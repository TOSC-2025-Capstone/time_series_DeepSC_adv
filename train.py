import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
from models.transceiver import DeepSC
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pdb
from tqdm import tqdm
import matplotlib.pyplot as plt

import os
import numpy as np
import pandas as pd
import pickle

from parameters.parameters import TrainParams, LossType, channel_type
import parameters.parameters as p
import csv
import time
from utils import log_epoch_stats_csv, plot_training_logs, train_mi, Channels, PowerNormalize
from models.mutual_info import sample_batch, mutual_information

def normalize_snr_to_range(snr_db):
    """SNR을 -2~2 범위로 정규화 (Clean=3)"""
    if snr_db == 100:
        return 3.0
    else:
        normalized = (snr_db - 3) / 18 * 4 - 2
        return normalized

def add_snr_label_to_tensor(tensor_data, is_train_data=True):
    """텐서에 SNR 레이블 추가"""
    batch_size, seq_len, features = tensor_data.shape

    main_features = tensor_data[:, :, :-1]
    file_index = tensor_data[:, :, -1:]

    clean_snr_label = torch.full((batch_size, seq_len, 1), 3.0)
    clean_with_label = torch.cat([main_features, clean_snr_label, file_index], dim=-1)

    augmented = clean_with_label

    if is_train_data:
        noise_label = torch.zeros(batch_size, 1, 1)
        noisy_snr_label = noise_label.expand(batch_size, seq_len, 1)
        noisy_with_label = torch.cat([main_features, noisy_snr_label, file_index], dim=-1)
        augmented = torch.cat([clean_with_label, noisy_with_label], dim=0)

    return augmented

def prepare_data_loaders(tensor_data, batch_size, is_train=True):
    """Clean/Noisy DataLoader 생성"""
    tensor_augmented = add_snr_label_to_tensor(tensor_data, is_train_data=True)

    snr_labels = tensor_augmented[:, 0, -2]
    clean_mask = (snr_labels == 3)
    noisy_mask = (snr_labels != 3)

    clean_data = tensor_augmented[clean_mask].clone()
    noisy_data = tensor_augmented[noisy_mask].clone()

    print(f"{'Train' if is_train else 'Val'} - Clean: {len(clean_data)}, Noisy: {len(noisy_data)}")

    clean_loader = DataLoader(
        TensorDataset(clean_data),
        batch_size=batch_size,
        shuffle=is_train,
        drop_last=is_train
    )

    noisy_loader = DataLoader(
        TensorDataset(noisy_data),
        batch_size=batch_size,
        shuffle=is_train,
        drop_last=is_train
    )

    return clean_loader, noisy_loader

def prepare_fixed_batch(val_tensor, device):
    """Fixed batch 준비 (SNR: 3dB, 12dB, 21dB)"""
    total_val_samples = len(val_tensor)
    sample_indices = [
        total_val_samples // 4,
        total_val_samples // 2,
        total_val_samples * 3 // 4
    ]

    fixed_snr_values = [3, 12, 21]
    fixed_samples = []

    for idx, snr_db in zip(sample_indices, fixed_snr_values):
        sample = val_tensor[idx].unsqueeze(0)
        main_features = sample[:, :, :-1]
        file_index = sample[:, :, -1:]

        snr_normalized = normalize_snr_to_range(snr_db)
        snr_label = torch.full((1, sample.size(1), 1), snr_normalized)

        sample_with_snr = torch.cat([main_features, snr_label, file_index], dim=-1)
        fixed_samples.append(sample_with_snr)

    fixed_batch = torch.cat(fixed_samples, dim=0).to(device)
    fixed_batch = fixed_batch[:, :, :-1]  # file_index 제거

    print("\nFixed batch SNR labels:")
    for i, snr in enumerate(fixed_snr_values):
        print(f"  Sample {i}: {snr}dB (normalized: {normalize_snr_to_range(snr):.2f})")

    return fixed_batch, fixed_snr_values

def process_batch(batch, device, is_clean, model, criterion, optimizer=None, is_training=True):
    """
    배치 처리 (Clean 또는 Noisy)

    Args:
        batch: 입력 배치
        device: cuda/cpu
        is_clean: Clean 배치 여부
        model: 모델
        criterion: 손실 함수
        optimizer: 옵티마이저 (학습 시)
        is_training: 학습 모드 여부

    Returns:
        loss: 손실값
    """
    batch = batch[0].to(device) if isinstance(batch, tuple) else batch.to(device)

    # SNR 레이블 확인
    snr_labels = batch[:, 0, -2]
    if is_clean:
        assert (snr_labels == 3).all(), "Clean batch에 noisy 샘플 섞임"
    else:
        assert (snr_labels != 3).all(), "Noisy batch에 clean 샘플 섞임"

    # 차원 처리
    batch = batch[:, :, :-1]  # file_index 제거 [batch, seq, 9]
    data_8d = batch[:, :, :-1]  # SNR 제거 [batch, seq, 8]

    if optimizer and is_training:
        optimizer.zero_grad()

    if is_clean:
        # Clean 처리
        p.is_train_phase = True
        output = model(batch)
        # output = model(data_8d) # 1126 snr 제거한 8개 피쳐학습
    else:
        # Noisy 처리
        snr_db = (np.random.randint(1, 8)) * 3  # 3~21
        # snr_db = 1 * 3  # 3~21
        model.snr_db = snr_db
        snr_normalized = normalize_snr_to_range(snr_db)

        # SNR 레이블 교체
        noisy_snr_label = torch.full(
            (batch.size(0), batch.size(1), 1),
            snr_normalized
        ).to(device)

        batch = batch[:, :, :-1]  # 기존 SNR 제거
        batch = torch.cat([batch, noisy_snr_label], dim=-1)  # 새 SNR 추가

        p.is_train_phase = False
        output = model(batch)

    # 손실 계산
    # output_8d = output # 1126
    output_8d = output[:, :, :-1] # SNR 제거
    # output_8d = torch.clamp(output_8d, min=0.0)  # 음수값 0으로 클램핑
    # output_8d = output_8d / 6  # minmax 스케일러 사용 시 6으로 나누기
    loss = criterion(output_8d, data_8d)

    if optimizer and is_training:
        loss.backward()
        optimizer.step()

    return loss

def run_epoch(clean_loader, noisy_loader, model, criterion, device,
              optimizer=None, is_training=True, epoch_num=1, total_epochs=1):
    """

    한 에포크 실행 (학습 또는 검증)

    Args:
        clean_loader: Clean DataLoader
        noisy_loader: Noisy DataLoader
        model: 모델
        criterion: 손실 함수
        device: cuda/cpu
        optimizer: 옵티마이저 (학습 시만)
        is_training: 학습 모드 여부
        epoch_num: 현재 에포크 번호
        total_epochs: 전체 에포크 수

    Returns:
        avg_loss: 평균 손실
    """
    if is_training:
        model.train()
    else:
        model.eval()

    clean_iter = iter(clean_loader)
    noisy_iter = iter(noisy_loader)

    max_batches = min(len(clean_loader), len(noisy_loader))
    batch_size = clean_loader.batch_size

    mode = "Train" if is_training else "Val"
    pbar = tqdm(range(max_batches), desc=f"Epoch {epoch_num}/{total_epochs} [{mode}]")

    total_loss = 0

    context = torch.no_grad() if not is_training else torch.enable_grad()

    with context:
        for _ in pbar:
            # Clean batch
            try:
                clean_batch = next(clean_iter)[0]
            except StopIteration:
                clean_iter = iter(clean_loader)
                clean_batch = next(clean_iter)[0]

            loss_clean = process_batch(
                clean_batch, device, is_clean=True,
                model=model, criterion=criterion,
                optimizer=optimizer if is_training else None,
                is_training=is_training
            )

            # Noisy batch
            try:
                noisy_batch = next(noisy_iter)[0]
            except StopIteration:
                noisy_iter = iter(noisy_loader)
                noisy_batch = next(noisy_iter)[0]

            loss_noisy = process_batch(
                noisy_batch, device, is_clean=False,
                model=model, criterion=criterion,
                optimizer=optimizer if is_training else None,
                is_training=is_training
            )

            total_loss += (loss_clean.item() + loss_noisy.item()) * batch_size
            # total_loss += loss_clean.item() * batch_size

            pbar.set_postfix({
                "Loss_clean": f"{loss_clean.item():.4f}",
                "Loss_noisy": f"{loss_noisy.item():.4f}"
            })

    avg_loss = total_loss / (max_batches * batch_size * 2)
    return avg_loss

def visualize_fixed_batch(fixed_batch, fixed_snr_values, model, device,
                          epoch, save_fig_dir):
    """Fixed batch 시각화"""
    fixed_data_8d = fixed_batch[:, :, :-1]

    for idx in range(3):
        os.makedirs(save_fig_dir, exist_ok=True)

        current_snr_db = fixed_snr_values[idx]
        model.snr_db = current_snr_db
        p.is_train_phase = False

        single_sample = fixed_batch[idx].unsqueeze(0)
        sample_output = model(single_sample)

        input_norm = fixed_data_8d[idx].unsqueeze(0).detach().cpu().numpy()
        output_norm = sample_output[0, :, :-1].unsqueeze(0).detach().cpu().numpy()

        plt.figure(figsize=(15, 8))

        for i in range(8):
            plt.subplot(2, 4, i + 1)
            plt.ylim(-3, 3)

            plt.plot(input_norm[0, :, i], label="Input", color="blue", alpha=0.7)
            plt.plot(output_norm[0, :, i], label="Output", color="orange", alpha=0.7)
            plt.title(f"Feature {i+1}")
            plt.legend()
            plt.grid(True)

        plt.suptitle(
            f"Epoch {epoch+1}, Sample {idx+1}, SNR={current_snr_db}dB"
        )
        plt.tight_layout()
        plt.savefig(
            f"{save_fig_dir}_epoch{epoch+1}_snr{current_snr_db}db_sample{idx+1}.png",
            dpi=200
        )
        plt.close()

    p.is_train_phase = True

def train_model(model=None, params: TrainParams=None, device=None, mi_net=None):
    """메인 학습 함수"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model is None or params is None:
        print("model과 params를 전달해주세요!")
        return

    # 파라미터 추출
    train_pt = params.train_pt
    validate_pt = params.validate_pt
    scaler_path = params.scaler_path
    model_save_path = params.model_save_path
    num_epochs = params.num_epochs
    batch_size = params.batch_size
    lr = params.lr
    save_fig_dir = params.save_fig_dir

    # 1. 데이터 로드
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_pt = os.path.join(current_dir, train_pt.lstrip("./"))
    validate_pt = os.path.join(current_dir, validate_pt.lstrip("./"))
    scaler_path = os.path.join(current_dir, scaler_path.lstrip("./"))

    print(f"Loading from: {train_pt}")
    train_data = torch.load(train_pt)
    val_data = torch.load(validate_pt)
    train_tensor = train_data.tensors[0]
    val_tensor = val_data.tensors[0]
    scaler = joblib.load(scaler_path)

    # 2. DataLoader 준비
    clean_train_loader, noisy_train_loader = prepare_data_loaders(
        train_tensor, batch_size, is_train=True
    )

    val_clean_loader, val_noisy_loader = prepare_data_loaders(
        val_tensor, batch_size, is_train=False
    )

    # 3. Fixed batch 준비
    fixed_batch, fixed_snr_values = prepare_fixed_batch(val_tensor, device)

    # 4. 손실함수 및 옵티마이저
    if params.loss_type == LossType.MSE.value:
        criterion = nn.MSELoss()
        print("학습 로스로 MSE가 설정되었습니다.")
    elif params.loss_type == LossType.MAE.value:
        criterion = nn.L1Loss()
    elif params.loss_type == LossType.Huber.value:
        criterion = nn.HuberLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, verbose=True)

    best_val_loss = float("inf")
    os.makedirs(model_save_path, exist_ok=True)

    # 5. 로그 초기화
    epoch_times = []
    os.makedirs(save_fig_dir, exist_ok=True)
    csv_file = os.path.join(save_fig_dir, "epoch_stats.csv")

    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Time(sec)", "AvgTime(sec)",
                        "TrainLoss", "ValLoss", "BestValLoss", "LR"])

    # 6. 학습 루프
    for epoch in range(num_epochs):
        start_time = time.time()

        # 학습
        avg_train_loss = run_epoch(
            clean_train_loader, noisy_train_loader,
            model, criterion, device,
            optimizer=optimizer,
            is_training=True,
            epoch_num=epoch+1,
            total_epochs=num_epochs
        )
        print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.6f}")

        # 검증
        avg_val_loss = run_epoch(
            val_clean_loader, val_noisy_loader,
            model, criterion, device,
            optimizer=None,
            is_training=False,
            epoch_num=epoch+1,
            total_epochs=num_epochs
        )
        print(f"[Epoch {epoch+1}] Val Loss: {avg_val_loss:.6f}")

        # 스케줄러
        scheduler.step(avg_val_loss)

        # 모델 저장
        if avg_val_loss < best_val_loss:
            torch.save(model.state_dict(), model_save_path + "best.pth")
            best_val_loss = avg_val_loss
            print(f"[Best Val Epoch {epoch+1}] Best Val Loss: {best_val_loss:.6f}")

        # 로깅
        end_time = time.time()
        lr_now = optimizer.param_groups[0]["lr"]
        epoch_times = log_epoch_stats_csv(
            start_time, end_time, epoch + 1, epoch_times, csv_file,
            avg_train_loss, avg_val_loss, best_val_loss, lr_now
        )

        # 시각화
        # if epoch == 0 or (epoch+1) % 40 == 0:
        #     visualize_fixed_batch(
        #         fixed_batch, fixed_snr_values, model, device,
        #         epoch, save_fig_dir
        #     )

        # 진행 상황 출력
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.6f}")
        print(f"  Val Loss: {avg_val_loss:.6f}")
        print(f"  Best Val Loss: {best_val_loss:.6f}")
        print(f'  Learning Rate: {lr_now:.6f}\n')

    print("학습 완료!")
    plot_training_logs(csv_file, save_fig_dir)
