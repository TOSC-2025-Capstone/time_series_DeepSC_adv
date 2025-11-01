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
import csv
import time
from utils import log_epoch_stats_csv, plot_training_logs, train_mi, Channels, PowerNormalize
from models.mutual_info import sample_batch, mutual_information

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
    train_loader = DataLoader(train_tensor, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_tensor, batch_size=batch_size, shuffle=False)
    # 2-1. sample data fix
    fixed_batch = train_tensor[:6].to(device)

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

        # 학습 루프
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch in train_pbar:
            batch = batch.to(device)

            # mi_net이 있다면 먼저 학습시키고
            mi = None
            snr_db = 10
            if mi_net != None :
                mi = train_mi(model, mi_net, batch, snr_db, None, mi_opt, channel_type)

            # 그 다음 메인 모델 학습
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)  # 복원 구조에서는 output = batch가 목적

            # mi_net을 평가모드로 전환 후 model 학습 결과 loss에 가중치(lambda = 0.0009)곱한 loss_mine을 더함
            if mi_net is not None:
                enc_output = model.encoder(batch, src_mask=None)
                compressed = model.time_compressor(enc_output)
                channel_enc_output = model.channel_encoder(compressed)
                Tx_sig = PowerNormalize(channel_enc_output)

                if channel_type == 'AWGN':
                    Rx_sig = channels.AWGN(Tx_sig, snr_db)
                elif channel_type == 'rayleigh':
                    Rx_sig = channels.Rayleigh(Tx_sig, snr_db)
                elif channel_type == 'rician':
                    Rx_sig = channels.Rician(Tx_sig, snr_db)
                else:
                    raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

                mi_net.eval()
                joint, marginal = sample_batch(Tx_sig, Rx_sig)
                mi_lb, _, _ = mutual_information(joint, marginal, mi_net)
                loss_mine = -mi_lb
                loss = loss + 0.0009 * loss_mine

            loss.backward()

            # 그래디언트 클리핑 추가
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item() * batch.size(0)
            train_pbar.set_postfix({"Loss": f"{loss.item():.6f}"})

        avg_train_loss = total_loss / len(train_loader.dataset)
        print(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.6f}")

        # 검증
        model.eval()
        val_loss = 0
        with torch.no_grad():
            # batch [train_batch_size, target_length(reshaped cycle len), d_comp]
            for batch in val_loader:
                batch = batch.to(device)
                output = model(batch)
                loss = criterion(output, batch)
                val_loss += loss.item() * batch.size(0)
        avg_val_loss = val_loss / len(val_loader.dataset)
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

        # === 정규화된 입력과 output 비교 plot (3개 배치만) ===
        # 0, 40, 80
        # if (epoch + 1) % 40 == 0:
        if 1 == 2:
            os.makedirs(save_fig_dir, exist_ok=True)
            # batch: [batch_size, window, feature]
            sample_output = model(fixed_batch)
            input_norm = fixed_batch[:3, :, :6].detach().cpu().numpy()  # [3, window, 6]
            output_norm = (
                sample_output[:3, :, :6].detach().cpu().numpy()
            )  # [6, window, 6]
            for sample_idx in range(3):
                plt.figure(figsize=(15, 8))
                for i in range(input_norm.shape[2]):
                    plt.subplot(2, 3, i + 1)
                    plt.plot(
                        input_norm[sample_idx, :, i],
                        label="Input (norm)",
                        color="blue",
                        alpha=0.7,
                    )
                    plt.plot(
                        output_norm[sample_idx, :, i],
                        label="Output (norm)",
                        color="orange",
                        alpha=0.7,
                    )
                    plt.title(f"Feature {i+1}")
                    plt.legend()
                    plt.grid(True)
                plt.suptitle(
                    f"정규화 입력 vs Output (Epoch {epoch+1}, Sample {sample_idx+1})"
                )
                plt.tight_layout()
                plt.savefig(
                    f"{save_fig_dir}_epoch{epoch+1}_sample{sample_idx+1}.png", dpi=200
                )
                # plt.show()

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
