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
    """
    SNR을 -3~3 범위로 정규화
    - Clean (100) → 4.0
    - 3dB → -2.0
    - 6dB → -1.33
    - 9dB → -0.67
    - 12dB → 0.0
    - 15dB → 0.67
    - 18dB → 1.33
    - 21dB → 2.0
    """
    if snr_db == 100:  # Clean
        return 3.0
    else:
        # 3~21dB를 -2~2로 선형 매핑
        # y = (x - 3) / (21 - 3) * (2 - (-2)) + (-2)
        # y = (x - 3) / 18 * 4 - 2
        normalized = (snr_db - 3) / 18 * 4 - 2
        return normalized

def add_snr_label_to_tensor(tensor_data, is_train_data = True):
    """
    [batch, seq_len, 9] → [batch, seq_len, 10]
    마지막 피처(file_index)는 유지, SNR은 그 앞에 추가
    """
    batch_size, seq_len, features = tensor_data.shape

    # 1. 피처 분리: [batch, seq_len, 8], [batch, seq_len, 1]
    main_features = tensor_data[:, :, :-1]  # [batch, seq_len, 8] - 처음 8개 피처
    file_index = tensor_data[:, :, -1:]     # [batch, seq_len, 1] - 마지막 file_index

    # 2. Clean 버전 생성
    clean_snr_label = torch.full((batch_size, seq_len, 1), 3.0)  # 3 = clean
    clean_with_label = torch.cat([
        main_features,      # [batch, seq_len, 8]
        clean_snr_label,    # [batch, seq_len, 1]
        file_index          # [batch, seq_len, 1]
    ], dim=-1)  # [batch, seq_len, 10]

    augmented = clean_with_label

    if is_train_data == True :
        # 3. Noisy 버전 생성 (벡터화)
        # [batch, 1, 1] 0으로 라벨링해서 나중에 노이즈 낄 데이터인지 판단
        noise_label = torch.zeros(batch_size, 1, 1)

        # 시퀀스 길이만큼 확장
        noisy_snr_label = noise_label.expand(batch_size, seq_len, 1)  # [batch, seq_len, 1]

        noisy_with_label = torch.cat([
            main_features,      # [batch, seq_len, 8]
            noisy_snr_label,    # [batch, seq_len, 1]
            file_index          # [batch, seq_len, 1]
        ], dim=-1)  # [batch, seq_len, 10]

        # 4. 결합: [2*batch, seq_len, 10]
        augmented = torch.cat([clean_with_label, noisy_with_label], dim=0)

    return augmented

"""
# train_model

모델과 파라미터를 입력받아 학습을 진행하는 함수
"""

# 기본값으로 train parameter 셋을 그대로 입력함 , model, device만 전달
def train_model(
    model=None,
    params:TrainParams= None,
    device=None,
    mi_net=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model is None:
        print("model을 전달해주세요!")
        return

    if params is None:
        print("params를 전달해주세요!")
        return

    # 파라미터에서 필요한 값들 추출
    train_pt = params.train_pt
    validate_pt = params.validate_pt
    scaler_path = params.scaler_path
    model_save_path = params.model_save_path
    num_epochs = params.num_epochs
    batch_size = params.batch_size
    lr = params.lr
    save_fig_dir = params.save_fig_dir

    # 1. 데이터 로드 (절대 경로로 변환)
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

    # 2. DataLoader
    # 데이터 준비
    train_tensor_augmented = add_snr_label_to_tensor(train_tensor)

    # Clean/Noisy 분리
    snr_labels = train_tensor_augmented[:, 0, -2]  # SNR 레이블 추출

    clean_mask = (snr_labels == 3)
    noisy_mask = (snr_labels != 3)

    clean_data = train_tensor_augmented[clean_mask].clone()
    noisy_data = train_tensor_augmented[noisy_mask].clone()

    print(f"Clean samples: {len(clean_data)}")
    print(f"Noisy samples: {len(noisy_data)}")

    # 두 개의 DataLoader 생성
    clean_train_loader = DataLoader(
        TensorDataset(clean_data),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True  # 마지막 배치 크기 통일
    )

    noisy_train_loader = DataLoader(
        TensorDataset(noisy_data),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    # train_loader = DataLoader(train_tensor_9d, batch_size=batch_size, shuffle=True)
    # train_loader = DataLoader(train_tensor, batch_size=batch_size, shuffle=True)
    # Validation 데이터 처리 수정
    val_tensor_augmented = add_snr_label_to_tensor(val_tensor, is_train_data=True)  # ← True로 변경하여 Clean/Noisy 분리

    # Clean/Noisy 분리
    val_snr_labels = val_tensor_augmented[:, 0, -2]
    val_clean_mask = (val_snr_labels == 3)
    val_noisy_mask = (val_snr_labels != 3)

    val_clean_data = val_tensor_augmented[val_clean_mask].clone()
    val_noisy_data = val_tensor_augmented[val_noisy_mask].clone()

    # Validation용 DataLoader 생성
    val_clean_loader = DataLoader(
        TensorDataset(val_clean_data),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )

    val_noisy_loader = DataLoader(
        TensorDataset(val_noisy_data),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )
    # val_loader = DataLoader(val_tensor_9d, batch_size=batch_size, shuffle=False)
    # val_loader = DataLoader(val_tensor, batch_size=batch_size, shuffle=False)

    # # 2-1. sample data fix
    # fixed_batch = val_tensor[:3 * (512 // p.segment_length_n)].to(device) # 파일 3개
    # fixed_batch = fixed_batch[:, :, :-1] # 인덱스 제거

    # 2-1. Fixed data 샘플링 수정 (동떨어진 3개 샘플 선택)
    total_val_samples = len(val_tensor)
    sample_indices = [
        total_val_samples // 4,      # 25% 지점
        total_val_samples // 2,      # 50% 지점
        total_val_samples * 3 // 4   # 75% 지점
    ]

    # SNR 고정: 3dB, 12dB, 21dB
    fixed_snr_values = [3, 12, 21]

    fixed_samples = []
    for idx, snr_db in zip(sample_indices, fixed_snr_values):
        sample = val_tensor[idx].unsqueeze(0)  # [1, seq_len, features]

        # 수동으로 SNR 레이블 추가
        main_features = sample[:, :, :-1]  # [1, seq_len, 8] - 피처
        file_index = sample[:, :, -1:]     # [1, seq_len, 1] - file_index

        # SNR 정규화
        snr_normalized = normalize_snr_to_range(snr_db)
        snr_label = torch.full((1, sample.size(1), 1), snr_normalized)

        # 결합: [1, seq_len, 10] = [8 features] + [1 SNR] + [1 file_index]
        sample_with_snr = torch.cat([main_features, snr_label, file_index], dim=-1)

        fixed_samples.append(sample_with_snr)

    fixed_batch = torch.cat(fixed_samples, dim=0).to(device)  # [3, seq_len, 10]
    fixed_batch = fixed_batch[:, :, :-1]  # file_index 제거 -> [3, seq_len, 9]

    print("Fixed batch SNR labels:")
    print(f"  Sample 0: {fixed_snr_values[0]}dB (normalized: {normalize_snr_to_range(fixed_snr_values[0]):.2f})")
    print(f"  Sample 1: {fixed_snr_values[1]}dB (normalized: {normalize_snr_to_range(fixed_snr_values[1]):.2f})")
    print(f"  Sample 2: {fixed_snr_values[2]}dB (normalized: {normalize_snr_to_range(fixed_snr_values[2]):.2f})")

    # 3. 모델 초기화
    input_dim = train_tensor.shape[2]
    # model = return_model("deepsc") # 파라미터에서 가져온 모델

    # 4. 손실함수 및 옵티마이저
    if params.loss_type == LossType.MSE.value:
        criterion = nn.MSELoss()
        print("학습 로스로 MSE가 설정되었습니다.")
    elif params.loss_type == LossType.MAE.value:
        criterion = nn.L1Loss()
        print("학습 로스로 MAE가 설정되었습니다.")
    elif params.loss_type == LossType.Huber.value:
        criterion = nn.HuberLoss()
        print("학습 로스로 HuberLoss가 설정되었습니다.")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    mi_opt = None
    if mi_net != None :
        mi_opt = torch.optim.Adam(mi_net.parameters(), lr=1e-3)

    best_val_loss = float("inf")

    # === 여기에 추가 ===
    early_stop_patience = 12
    early_stop_counter = 0

    os.makedirs(model_save_path, exist_ok=True)

    # === epoch time 기록 리스트 ===
    epoch_times = []

    # === 로그 CSV 파일 경로 설정 ===
    os.makedirs(save_fig_dir, exist_ok=True)
    csv_file = os.path.join(save_fig_dir, "epoch_stats.csv")

    # CSV 헤더 초기화
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Time(sec)", "AvgTime(sec)",
                            "TrainLoss", "ValLoss", "BestValLoss", "LR"])

    for epoch in range(num_epochs):
        start_time = time.time()  # 시작 시각 기록
        channels = Channels()

        # 학습 모드
        model.train()
        total_loss = 0

        # Clean/Noisy 번갈아가며 학습
        clean_iter = iter(clean_train_loader)
        noisy_iter = iter(noisy_train_loader)

        max_batches = min(len(clean_train_loader), len(noisy_train_loader))

        train_pbar = tqdm(range(max_batches), desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in train_pbar:
            # === Clean batch 학습 ===
            try:
                clean_batch = next(clean_iter)[0].to(device)
            except StopIteration:
                clean_iter = iter(clean_train_loader)
                clean_batch = next(clean_iter)[0].to(device)

            # SNR 레이블 확인 (전부 100이어야 함)
            snr_labels = clean_batch[:, 0, -2]
            assert (snr_labels == 3).all(), "Clean batch에 noisy 샘플 섞임"

            # 10->9차원
            clean_batch = clean_batch[:,:,:-1] # file index 제거

            # loss 비교용 데이터 추출 (snr label값은 비교안함)
            data_8d = clean_batch[:, :, :-1]  # [batch, seq, 8] - SNR, file_index 제거

            optimizer.zero_grad()

            p.is_train_phase = True
            output = model(clean_batch) # 9차원 입력
            output_8d = output[:, :, :-1]  # 출력도 9차원이므로 8차원으로 변환
            loss_clean = criterion(output_8d, data_8d)
            loss_clean.backward()
            optimizer.step()

            # === Noisy batch 학습 ===
            try:
                noisy_batch = next(noisy_iter)[0].to(device)
            except StopIteration:
                noisy_iter = iter(noisy_train_loader)
                noisy_batch = next(noisy_iter)[0].to(device)

            # SNR 레이블 확인 (전부 같은 값이어야 함)
            snr_labels = noisy_batch[:, 0, -2]
            assert (snr_labels != 3).all(), "Noisy batch에 clean 샘플 섞임"

            # 10->9차원
            noisy_batch = noisy_batch[:, :, :-1] # file index 제거

            # loss 비교용 데이터 추출 (snr label값은 비교안함)
            n_data_8d = noisy_batch[:, :, :-1]
            optimizer.zero_grad()

            # 랜덤 SNR (3~21dB)
            snr_db = (np.random.randint(1, 7)) * 3
            model.snr_db = snr_db
            snr_normalized = normalize_snr_to_range(snr_db)

            noisy_snr_label = torch.full(
                    (noisy_batch.size(0), noisy_batch.size(1), 1),
                    snr_normalized  # ← 정규화된 값 사용
                ).to(device)
            noisy_batch = noisy_batch[:,:,:-1] # 기존에 있던 noisy label 제거
            noisy_batch = torch.cat([noisy_batch, noisy_snr_label], dim=-1) # 그 자리에 snr 라벨 추가

            p.is_train_phase = False
            output = model(noisy_batch) # 9차원 입력
            output_8d = output[:, :, :-1]  # 출력도 9차원이므로 8차원으로 변환
            loss_noisy = criterion(output_8d, n_data_8d)
            loss_noisy.backward()
            optimizer.step()

            # Loss 기록
            total_loss += (loss_clean.item() + loss_noisy.item()) * batch_size
            train_pbar.set_postfix({
                "Loss_clean": f"{loss_clean.item():.4f}",
                "Loss_noisy": f"{loss_noisy.item():.4f}"
            })

        avg_train_loss = total_loss / (max_batches * batch_size * 2) # 두 트레인 로더 배치 다합친 수 * 배치 사이즈 * 한 배치에 두번 학습
        print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.6f}")

        # ============= Validation  =============
        p.is_train_phase = True
        model.eval()
        val_loss = 0

        with torch.no_grad():
            val_clean_iter = iter(val_clean_loader)
            val_noisy_iter = iter(val_noisy_loader)

            max_val_batches = min(len(val_clean_loader), len(val_noisy_loader))

            val_pbar = tqdm(range(max_val_batches), desc=f"Epoch {epoch+1}/{num_epochs} [Val]")

            for _ in val_pbar:
                # === Clean validation ===
                try:
                    val_clean_batch = next(val_clean_iter)[0].to(device)
                except StopIteration:
                    val_clean_iter = iter(val_clean_loader)
                    val_clean_batch = next(val_clean_iter)[0].to(device)

                val_clean_batch = val_clean_batch[:, :, :-1]  # file_index 제거
                val_clean_data = val_clean_batch[:, :, :-1]   # SNR 제거 (8차원)

                p.is_train_phase = True
                clean_output = model(val_clean_batch)
                clean_output_8d = clean_output[:, :, :-1]
                loss_clean = criterion(clean_output_8d, val_clean_data)

                # === Noisy validation ===
                try:
                    val_noisy_batch = next(val_noisy_iter)[0].to(device)
                except StopIteration:
                    val_noisy_iter = iter(val_noisy_loader)
                    val_noisy_batch = next(val_noisy_iter)[0].to(device)

                val_noisy_batch = val_noisy_batch[:, :, :-1]  # file_index 제거
                val_noisy_data_orig = val_noisy_batch[:, :, :-1]  # SNR 제거 (8차원)

                # 랜덤 SNR
                snr_db = (np.random.randint(1, 8)) * 3
                model.snr_db = snr_db
                snr_normalized = normalize_snr_to_range(snr_db)

                noisy_snr_label = torch.full(
                    (val_noisy_batch.size(0), val_noisy_batch.size(1), 1),
                    snr_normalized
                ).to(device)

                val_noisy_batch = val_noisy_batch[:, :, :-1]  # 기존 SNR 제거
                val_noisy_batch = torch.cat([val_noisy_batch, noisy_snr_label], dim=-1)

                p.is_train_phase = False
                noisy_output = model(val_noisy_batch)
                noisy_output_8d = noisy_output[:, :, :-1]
                loss_noisy = criterion(noisy_output_8d, val_noisy_data_orig)

                # Loss 누적
                val_loss += (loss_clean.item() + loss_noisy.item()) * batch_size

        avg_val_loss = val_loss / (max_val_batches * batch_size * 2)
        print(f"[Epoch {epoch+1}/{num_epochs}] Val Loss: {avg_val_loss:.6f}")

        # 스케줄러 step (val loss 기준)
        scheduler.step(avg_val_loss)

        # val loss 개선 시 모델 저장
        if avg_val_loss < best_val_loss:
            torch.save(model.state_dict(), model_save_path + "best.pth")
            best_val_loss = avg_val_loss
            best_epoch_idx = epoch
            print(
                f"[Best Val Epoch {epoch+1}/{num_epochs}] Best Val Loss: {best_val_loss}"
            )

        # === epoch 시간 측정 및 로깅 (CSV 저장) ===
        end_time = time.time()
        lr_now = optimizer.param_groups[0]["lr"]
        epoch_times = log_epoch_stats_csv(
            start_time, end_time, epoch + 1, epoch_times, csv_file,
            avg_train_loss, avg_val_loss, best_val_loss, lr_now
        )

        # ============= Fixed batch 시각화 수정 (epoch % 40 == 0) =============
        if epoch == 0 or (epoch+1) % 40 == 0:
            # Fixed batch는 [3, seq_len, 9] 형태
            fixed_data_8d = fixed_batch[:, :, :-1]  # [3, seq_len, 8] - SNR 제거

            # 각 샘플의 SNR 설정
            fixed_snr_values = [3, 12, 21]
            fixed_snr_labels = [normalize_snr_to_range(snr) for snr in fixed_snr_values]

            for idx in range(3):
                os.makedirs(save_fig_dir, exist_ok=True)

                # 해당 샘플의 SNR 설정
                current_snr_db = fixed_snr_values[idx]
                model.snr_db = current_snr_db
                p.is_train_phase = False  # Noisy 모드

                # 단일 샘플 처리
                single_sample = fixed_batch[idx].unsqueeze(0)  # [1, seq_len, 9]
                sample_output = model(single_sample)  # [1, seq_len, 9]

                # 입력/출력 8차원 변환
                input_norm = fixed_data_8d[idx].unsqueeze(0).detach().cpu().numpy()  # [1, seq_len, 8]
                output_norm = sample_output[0, :, :-1].unsqueeze(0).detach().cpu().numpy()  # [1, seq_len, 8]

                plt.figure(figsize=(15, 8))

                for i in range(8):  # 8개 피처
                    plt.subplot(2, 4, i + 1)
                    plt.ylim(-3, 3)

                    plt.plot(
                        input_norm[0, :, i],
                        label="Input (norm)",
                        color="blue",
                        alpha=0.7,
                    )
                    plt.plot(
                        output_norm[0, :, i],
                        label="Output (norm)",
                        color="orange",
                        alpha=0.7,
                    )
                    plt.title(f"Feature {i+1}")
                    plt.legend()
                    plt.grid(True)

                plt.suptitle(
                    f"정규화 입력 vs Output (Epoch {epoch+1}, Sample {idx+1}, SNR={current_snr_db}dB)"
                )
                plt.tight_layout()
                plt.savefig(
                    f"{save_fig_dir}_epoch{epoch+1}_sample{idx+1}_snr{current_snr_db}db.png", dpi=200
                )
                plt.close()

            # 모드 복원
            p.is_train_phase = True

        # 진행 상황 출력
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.6f}")
        print(f"  Val Loss: {avg_val_loss:.6f}")
        print(f"  Best Val Loss: {best_val_loss:.6f}")
        print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
    print("학습 완료!")

    # === 학습 로그 시각화 ===
    plot_training_logs(csv_file, save_fig_dir)


# if __name__ == "__main__":
#     train_model()
