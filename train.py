import os
import torch
import torch.nn as nn
import torch.optim as optim
# [수정] Dataset 클래스를 사용하기 위해 import
from torch.utils.data import DataLoader, TensorDataset, Dataset
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

# [수정] random import (랜덤 크롭을 위해)
import random

"""
# [신규] CycleSegmentDataset

논의한 2번 방식(동적 분할)을 구현하는 커스텀 Dataset 클래스입니다.
- __init__: (사이클 수, target_length, 5) 텐서와 분할할 길이 n을 받습니다.
- __len__: 총 사이클 수를 반환합니다.
- __getitem__:
    - (학습 시) 사이클 내에서 n 길이의 랜덤한 세그먼트를 반환합니다.
    - (검증 시) 사이클 내에서 n 길이의 고정된 첫 세그먼트를 반환합니다.
"""
class CycleSegmentDataset(Dataset):
    def __init__(self, full_cycles_tensor, segment_length_n, is_train=True):
        """
        Args:
            full_cycles_tensor (torch.Tensor): (num_cycles, target_length, num_features)
            segment_length_n (int): 분할할 시퀀스 길이 (n)
            is_train (bool): True이면 랜덤 크롭, False이면 고정 크롭
        """
        self.data = full_cycles_tensor
        self.n = segment_length_n
        self.is_train = is_train

        # 원본 사이클의 전체 길이
        self.target_length = self.data.shape[1]

        if self.target_length < self.n:
            # 이 경우 패딩이 필요하지만, 전처리에서 target_length를 고정했으므로
            # n이 target_length보다 크지 않다고 가정합니다.
            raise ValueError(f"segment_length_n(n={self.n})이 "
                             f"원본 시퀀스 길이(target_length={self.target_length})보다 큽니다.")

    def __len__(self):
        # 데이터셋의 크기 = 총 사이클의 수
        return self.data.shape[0]

    def __getitem__(self, idx):
        # idx에 해당하는 전체 사이클 데이터를 가져옴
        # shape: (target_length, num_features)
        full_cycle = self.data[idx]

        if self.is_train:
            # [학습] 랜덤 크롭: 0 ~ (전체길이 - n) 사이에서 랜덤 시작점 선택
            start_index = random.randint(0, self.target_length - self.n)
        else:
            # [검증] 고정 크롭: 항상 맨 앞부터 n개 선택
            start_index = 0

        # [n, num_features] 형태의 세그먼트 반환
        segment = full_cycle[start_index : start_index + self.n]

        return segment


"""
# train_model현재 사용중인 디바이스

모델과 파라미터를 입력받아 학습을 진행하는 함수
(DataLoader 부분이 CycleSegmentDataset을 사용하도록 수정됨)
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

    try:
        segment_length_n = params.segment_length_n
    except AttributeError:
        print("경고: TrainParams에 'segment_length_n'이 없습니다. 기본값 5를 사용합니다.")
        segment_length_n = 8 # n의 기본값

    # 1. 데이터 로드 (절대 경로로 변환)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_pt = os.path.join(current_dir, train_pt.lstrip("./"))
    validate_pt = os.path.join(current_dir, validate_pt.lstrip("./"))
    scaler_path = os.path.join(current_dir, scaler_path.lstrip("./"))

    print(f"Loading from: {train_pt}")
    train_data = torch.load(train_pt)
    val_data = torch.load(validate_pt)

    # .pt 파일에 저장된 (N_cycles, target_length, N_features) 텐서 추출
    train_tensor = train_data.tensors[0]
    val_tensor = val_data.tensors[0]
    scaler = joblib.load(scaler_path)

    print(f"Train tensor shape: {train_tensor.shape}") # (Cycles, TargetLength, Features)
    print(f"Val tensor shape: {val_tensor.shape}")   # (Cycles, TargetLength, Features)

    # [수정] 2. Dataset 및 DataLoader (CycleSegmentDataset 사용)
    train_dataset = CycleSegmentDataset(
        train_tensor,
        segment_length_n,
        is_train=True
    )
    val_dataset = CycleSegmentDataset(
        val_tensor,
        segment_length_n,
        is_train=False # 검증 시에는 랜덤 크롭 X
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4 # 데이터 로딩 속도 향상 (선택 사항)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    # [수정] 2-1. sample data fix (검증 데이터셋에서 6개 샘플 고정)
    # val_dataset은 항상 0번 인덱스부터 n개만큼 자르므로 결과가 고정됨
    num_fixed_samples = min(6, len(val_dataset))
    fixed_batch_list = [val_dataset[i] for i in range(num_fixed_samples)]
    fixed_batch = torch.stack(fixed_batch_list).to(device)
    # fixed_batch shape: (6, n, num_features)

    # 3. 모델 초기화
    input_dim = train_tensor.shape[2] # 피쳐 수는 동일
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
        # 이제 train_loader는 (batch_size, n, num_features) 형태의 배치를 반환
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch in train_pbar:
            # batch shape: [batch_size, segment_length_n, num_features]
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

        # [수정] len(train_loader.dataset)은 이제 총 '사이클 수'입니다.
        # 따라서 avg_train_loss 계산이 약간 달라질 수 있으나,
        # CycleSegmentDataset이 1 사이클당 1 세그먼트(랜덤)를 반환하므로
        # 1 에포크당 총 사이클 수만큼의 세그먼트를 보는 것은 동일합니다.
        avg_train_loss = total_loss / len(train_loader.dataset)
        print(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.6f}")

        # 검증
        model.eval()
        val_loss = 0
        with torch.no_grad():
            # val_loader는 (batch_size, n, num_features) 형태의 고정된 배치를 반환
            for batch in val_loader:
                batch = batch.to(device)
                output = model(batch)
                loss = criterion(output, batch)
                val_loss += loss.item() * batch.size(0)

        # len(val_loader.dataset)은 총 '검증 사이클 수'
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
        if (epoch + 1) % 40 == 0:
        # if 1 == 2:
            os.makedirs(save_fig_dir, exist_ok=True)
            # batch: [batch_size, window, feature]
            # fixed_batch는 이미 (6, n, 5) 형태이므로 바로 사용 가능
            sample_output = model(fixed_batch)

            # fixed_batch의 shape[0] (샘플 수)가 3보다 작을 수 있으므로 min 사용
            num_plot_samples = min(3, fixed_batch.shape[0])

            # (num_plot_samples, n, num_features)
            input_norm = fixed_batch[:num_plot_samples, :, :].detach().cpu().numpy()
            output_norm = (
                sample_output[:num_plot_samples, :, :].detach().cpu().numpy()
            )

            for sample_idx in range(num_plot_samples):
                plt.figure(figsize=(15, 8))

                # 피쳐 수 (input_norm.shape[2]) 만큼 반복
                num_features_to_plot = min(6, input_norm.shape[2]) # 최대 6개 피쳐만

                for i in range(num_features_to_plot):
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
                plt.close() # 메모리 관리를 위해 플롯 닫기

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
#     # 이 파일을 직접 실행할 경우 TrainParams 객체를 생성하고
#     # segment_length_n 값을 설정한 뒤 train_model을 호출해야 합니다.
#     # 예:
#     # my_params = TrainParams()
#     # my_params.segment_length_n = 100
#     # model = ... (모델 로드)
#     # train_model(model=model, params=my_params)
#     pass
