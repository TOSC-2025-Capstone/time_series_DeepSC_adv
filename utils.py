# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 09:47:54 2020

@author: HQ Xie
utils.py
"""
import os
import math
import torch
import time
import torch.nn as nn
from models.mutual_info import sample_batch, mutual_information
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdb

import csv
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import matplotlib.pyplot as plt
import numpy as np
import torch

def visualize_semantic_constellation(tx_sig, rx_sig, snr_db):
    """
    [16, 4, 2] 형태의 텐서를 받아 성상도를 그립니다.
    tx_sig: Channel Encoder 출력 (Batch, Time, Dim=2)
    rx_sig: Rayleigh 채널 및 등화(Equalization) 후 출력 (Batch, Time, Dim=2)
    """
    # 1. 데이터 준비 (CPU변환 및 numpy)
    # shape: [16, 4, 2] -> [64, 2]로 펼치기 전에, 시간 축 정보를 유지하기 위해 분리
    batch_size, time_steps, dim = tx_sig.shape

    tx_data = tx_sig.detach().cpu().numpy()
    rx_data = rx_sig.detach().cpu().numpy()

    # 색상맵 (시간 스텝 0, 1, 2, 3을 구분하기 위함)
    colors = plt.cm.viridis(np.linspace(0, 1, time_steps))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # --- 첫 번째 그래프: 송신 신호 (Tx) - 모델이 학습한 성상도 ---
    ax1 = axes[0]
    for t in range(time_steps):
        # 모든 배치의 t번째 시간 스텝만 추출 [16, 2]
        tx_points = tx_data[:, t, :]
        ax1.scatter(tx_points[:, 0], tx_points[:, 1],
                   color=colors[t], label=f'Time Step {t}', s=50, alpha=0.8, edgecolors='w')

    ax1.set_title(f'Learned Semantic Constellation (Tx)\n(Color by Compressed Time Step)', fontsize=14)
    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.axvline(0, color='black', linewidth=0.5)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    # --- 두 번째 그래프: 수신 신호 (Rx) - 채널 영향 ---
    ax2 = axes[1]
    # 비교를 위해 Tx를 희미하게 깔아줌
    flat_tx = tx_data.reshape(-1, 2)
    ax2.scatter(flat_tx[:, 0], flat_tx[:, 1], color='blue', alpha=0.1, s=20, label='Tx Centers')

    # Rx 데이터 플롯
    flat_rx = rx_data.reshape(-1, 2)
    ax2.scatter(flat_rx[:, 0], flat_rx[:, 1], color='red', alpha=0.4, s=20, label='Rx (Noisy)')

    ax2.set_title(f'Received Signals after Rayleigh Fading & EQ\n(SNR: {snr_db}dB)', fontsize=14)
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.axvline(0, color='black', linewidth=0.5)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    # 축 범위 통일 (최대값 기준)
    max_val = max(np.abs(flat_tx).max(), np.abs(flat_rx).max()) * 1.1
    for ax in axes:
        ax.set_xlim(-max_val, max_val)
        ax.set_ylim(-max_val, max_val)
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.show()


class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        # 将数组全部填充为某一个值
        true_dist.fill_(self.smoothing / (self.size - 2))
        # 按照index将input重新排列
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        # 第一行加入了<strat> 符号，不需要加入计算
        true_dist[:, self.padding_idx] = 0 #
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist)


class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        self._weight_decay = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        weight_decay = self.weight_decay()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
            p['weight_decay'] = weight_decay
        self._rate = rate
        self._weight_decay = weight_decay
        # update weights
        self.optimizer.step()

    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step

        # if step <= 3000 :
        #     lr = 1e-3

        # if step > 3000 and step <=9000:
        #     lr = 1e-4

        # if step>9000:
        #     lr = 1e-5

        lr = self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

        return lr


        # return lr

    def weight_decay(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step

        if step <= 3000 :
            weight_decay = 1e-3

        if step > 3000 and step <=9000:
            weight_decay = 0.0005

        if step>9000:
            weight_decay = 1e-4

        weight_decay =   0
        return weight_decay

def power_normalize(x):
    x_square = torch.mul(x, x)
    power = torch.mean(x_square).sqrt()
    # power = (2 * torch.mean(x_square)).sqrt()
    x = torch.div(x, power)
    return x

class Channels():
    def AWGN(self, Tx_sig, snr_db=10):
        snr_linear = 10 ** (snr_db / 10)          # dB → 선형 변환 (신호 전력/노이즈 전력 값)
        signal_power = Tx_sig.pow(2).mean().item()  # 신호 평균 전력
        noise_std = 1 / np.sqrt(2 * snr_linear)          # 잡음 분산 -> 노이즈 전력값
        # noise_std = math.sqrt(n_var)
        print("noise_std:", noise_std, "signal power:", signal_power)
        # print("noise variance:", n_var, "signal power:", signal_power, "Tx sig :" , Tx_sig[0,:5])

        noise = torch.normal(
            mean=0,
            std=noise_std,
            # std=0.66,
            size=Tx_sig.shape,
            device=Tx_sig.device
        )
        # print("mean:",torch.mean(noise),' std:' ,torch.std(noise), math.sqrt(n_var), n_var, "sigpower:", signal_power)

        # 데이터 준비
        tx_signals = torch.flatten(Tx_sig, 0, 1).detach().cpu().numpy()
        noise_signals = torch.flatten(noise, 0, 1).detach().cpu().numpy()
        # rx_signals = [tx_signals[i] + noise_signals[i] for i in range(3)]
        rx_signals = tx_signals + noise_signals

        # 전체 데이터의 min/max 계산 (통일된 스케일)
        all_data = tx_signals + noise_signals + rx_signals
        y_min = min(data.min() for data in all_data)
        y_max = max(data.max() for data in all_data)
        x_range = [0, 128]

        # 3x3 그리드 생성
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        feature_names = ['Feature 0', 'Feature 1']
        column_titles = ['Tx Signal', 'Noise', 'Rx Signal']

        # 각 서브플롯 그리기
        for row in range(2):  # 피처 (0, 1, 2)
            for col in range(3):  # Tx, Noise, Rx
                ax = axes[row, col]

                # 데이터 선택
                if col == 0:  # Tx
                    data = tx_signals[:, row]
                    color = 'blue'
                elif col == 1:  # Noise
                    data = noise_signals[:, row]
                    color = 'orange'
                else:  # Rx
                    data = rx_signals[:, row]
                    color = 'red'
                # 플롯
                ax.plot(data, color=color, linewidth=1.5, alpha=0.8)

                # 스케일 통일
                ax.set_xlim(x_range)
                ax.set_ylim([y_min, y_max])

                # 그리드
                ax.grid(True, alpha=0.3)

                for x_pos in [i*8 for i in range(1,17)]:
                    ax.axvline(x=x_pos, color='r', linestyle='--', label=f'Vertical line at {x_pos}')

                # 라벨 (첫 행에만 열 제목, 첫 열에만 행 제목)
                if row == 0:
                    ax.set_title(column_titles[col], fontsize=12, fontweight='bold')
                if col == 0:
                    ax.set_ylabel(feature_names[row], fontsize=11, fontweight='bold')

                # x축 라벨 (마지막 행에만)
                if row == 1:
                    ax.set_xlabel('Time Step', fontsize=10)

        # 전체 타이틀
        fig.suptitle(f'Signal Comparison (SNR={snr_db}dB)\n'
                    f'Signal Power: {signal_power:.6f}, Noise Std: {noise_std:.6f}',
                    fontsize=14, fontweight='bold', y=0.995)

        plt.tight_layout()
        plt.savefig('signal_comparison_3x3.png', dpi=300, bbox_inches='tight')
        plt.show()

        # self.visualize_awgn_effect(Tx_sig, snr_db=15)
        return Tx_sig + noise

    def Rayleigh(self, Tx_sig, snr=10):
        shape = Tx_sig.shape
        H_real = torch.normal(0, math.sqrt(1/2), size=[1]).to(device)
        H_imag = torch.normal(0, math.sqrt(1/2), size=[1]).to(device)
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(device)
        Tx_sig = torch.matmul(Tx_sig.view(shape[0], -1, 2), H) # 16, n, 2 * 2, 2
        Rx_sig = self.AWGN(Tx_sig, snr)
        # visualize_cycle_performance(Tx_sig[0], Rx_sig[0])
        Rx_sig_orig = Rx_sig
        # Channel estimation
        Rx_sig = torch.matmul(Rx_sig, torch.inverse(H)).view(shape)
        # visualize_cycle_performance(Rx_sig[0], Rx_sig_orig[0])

        return Rx_sig

    def Rician(self, Tx_sig, snr=10, K=1):
        shape = Tx_sig.shape
        mean = math.sqrt(K / (K + 1))
        std = math.sqrt(1 / (K + 1))
        H_real = torch.normal(mean, std, size=[1]).to(device)
        H_imag = torch.normal(mean, std, size=[1]).to(device)
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(device)
        Tx_sig = torch.matmul(Tx_sig.view(shape[0], -1, 2), H) # 16, 8, 2 * 2, 2
        Rx_sig = self.AWGN(Tx_sig, snr)
        # Channel estimation
        Rx_sig = torch.matmul(Rx_sig, torch.inverse(H)).view(shape)

        return Rx_sig

def visualize_cycle_performance(tx_df, rx_df):
        feature_cols = []
        # 1. 원본-복원 비교 플롯
        print(tx_df.shape)
        plt.figure(figsize=(18, 10))
        for i in range(2):
            plt.subplot(2, 4, i + 1)
            plt.plot(tx_df[: , i].cpu(), label="x", alpha=0.7)
            plt.plot(rx_df[: , i].cpu(), label="final", alpha=0.7)
            plt.legend()
            plt.ylim([-3,3])
            plt.grid(True)
            # for x_pos in [i*16 for i in range(1,33)]:
            #     plt.axvline(x=x_pos, color='r', linestyle='--', label=f'Vertical line at {x_pos}')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

# class Channels(nn.Module):
#     def __init__(self):
#         """
#         args 대신 is_train_phase 플래그를 직접 받아 초기화합니다.
#         is_train_phase=True: 학습 모드 (노이즈 X - 원본 TF 코드 로직 기준)
#         is_train_phase=False: 추론/검증 모드 (노이즈 O)
#         """
#         super(Channels, self).__init__()
#         # is_train_phase = is_train_phase

#     def awgn(self, inputs, n_std=0.1):
#         """
#         AWGN 채널을 시뮬레이션합니다.
#         """
#         x = inputs
#         y = None

#         # torch.randn_like(x)는 평균 0, 표준편차 1의 노이즈를 생성
#         # n_std를 곱해 표준편차를 조절
#         # 입력 텐서와 동일한 device 및 dtype을 사용해야 함
#         noise = torch.randn_like(x) * n_std
#         y = x + noise

#         return y

#     def fading(self, inputs, K=1, n_std=0.1, detector="MMSE"):
#         """
#         Fading 채널(Rician) 및 검출기(LS, MMSE)를 시뮬레이션합니다.
#         마찬가지로 is_train_phase=False일 때만 AWGN 노이즈(n)를 추가합니다.
#         """
#         pdb.set_trace()
#         x = inputs
#         # PyTorch에서는 .shape 대신 .size() 또는 .shape 사용
#         bs, sent_len, d_model = x.shape

#         # K 값은 Python float이므로 math 사용 유지
#         mean = math.sqrt(K / (2 * (K + 1)))
#         std = math.sqrt(1 / (2 * (K + 1)))

#         # tf.reshape -> x.view (더 효율적)
#         # tf.complex(x_real, x_imag)

#         # 원본 TF 코드는 [bs, -1, 2]로 변환합니다.
#         # 이는 (sent_len * d_model / 2) 길이의 복소수 벡터를 의미합니다.
#         # [bs, sent_len, d_model] -> [bs, -1, 2]
#         x_reshaped = x.reshape(bs, -1, 2) # 실수부와 허수부 2개의 피쳐로 구분할 수 있게 쉐이핑
#         x_complex = torch.complex(x_reshaped[..., 0], x_reshaped[..., 1]) # [bs, (sent_len*d_model)/2]

#         # h 채널 생성 (Rician)
#         # tf.random.normal((1,), ...) -> torch.normal(mean, std, size=(1,))
#         # h가 배치 전체에 동일하게 적용되도록 브로드캐스팅 활용
#         # 입력 텐서와 동일한 device 및 dtype 사용
#         h_real = torch.normal(mean, std, size=(1,), device=x.device, dtype=torch.float32)
#         h_imag = torch.normal(mean, std, size=(1,), device=x.device, dtype=torch.float32)
#         h_complex = torch.complex(h_real, h_imag) # shape: [1]

#         # 노이즈 벡터 생성 (AWGN)
#         # x_complex와 동일한 shape의 복소수 노이즈 생성
#         # 복소수 노이즈의 표준편차가 n_std가 되도록 실수/허수부 표준편차를 n_std/sqrt(2)로 설정
#         n_std_complex = n_std / math.sqrt(2)
#         n_real = torch.normal(0.0, n_std_complex, size=x_complex.shape, device=x.device, dtype=torch.float32)
#         n_imag = torch.normal(0.0, n_std_complex, size=x_complex.shape, device=x.device, dtype=torch.float32)
#         n_complex = torch.complex(n_real, n_imag) # [bs, (sent_len*d_model)/2]

#         # y = hx + n
#         # y_complex = None
#         y_complex = x_complex * h_complex + n_complex

#         # Detector (Perfect CSI)
#         x_est_complex = None
#         if detector == "LS":
#             # h_complex_conj = tf.math.conj(h_complex)
#             h_complex_conj = torch.conj(h_complex)
#             # 분모가 0이 되는 것을 방지하기 위해 작은 값(epsilon) 추가
#             eps = 1e-10
#             x_est_complex = y_complex * h_complex_conj / (h_complex * h_complex_conj + eps)
#         elif detector == "MMSE":
#             # h_complex_conj = tf.math.conj(h_complex)
#             h_complex_conj = torch.conj(h_complex)

#             # 원본 TF 코드의 (n_std * n_std * 2)는 실수부/허수부 분산의 합을 의미 (각각 n_std^2)
#             # PyTorch에서는 복소수 노이즈의 총 분산 E[|n|^2] = E[n_r^2] + E[n_i^2] = (n_std_complex^2) + (n_std_complex^2) = n_std^2
#             noise_power = n_std * n_std

#             a = (h_complex * h_complex_conj) + noise_power
#             x_est_complex = y_complex * h_complex_conj / a
#         else:
#             raise ValueError("detector must be in ['LS', 'MMSE']")

#         # x_est를 다시 [bs, sent_len, d_model] 형태로 복원
#         # x_est_real = tf.math.real(x_est_complex)
#         # x_est_img = tf.math.imag(x_est_complex)
#         x_est_real = torch.real(x_est_complex)
#         x_est_img = torch.imag(x_est_complex)

#         # x_est_real = tf.expand_dims(x_est_real, -1)
#         # x_est_img = tf.expand_dims(x_est_img, -1)
#         x_est_real = x_est_real.unsqueeze(-1)
#         x_est_img = x_est_img.unsqueeze(-1)

#         # x_est = tf.concat([x_est_real, x_est_img], axis=-1)
#         x_est = torch.cat((x_est_real, x_est_img), dim=-1) # [bs, (sent_len*d_model)/2, 2]

#         # x_est = tf.reshape(x_est, (bs, sent_len, -1))
#         x_est = x_est.reshape(bs, sent_len, -1) # [bs, sent_len, d_model]

#         # 원본 TF 코드의 out1, out2는 반환되지 않으므로 x_est만 반환
#         return x_est



def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    # 产生下三角矩阵
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask)

def create_masks(src, trg, padding_idx):

    src_mask = (src == padding_idx).unsqueeze(-2).type(torch.FloatTensor) #[batch, 1, seq_len]

    trg_mask = (trg == padding_idx).unsqueeze(-2).type(torch.FloatTensor) #[batch, 1, seq_len]
    look_ahead_mask = subsequent_mask(trg.size(-1)).type_as(trg_mask.data)
    combined_mask = torch.max(trg_mask, look_ahead_mask)

    return src_mask.to(device), combined_mask.to(device)

def PowerNormalize(x):
    x_square = torch.mul(x, x)
    power = torch.mean(x_square).sqrt()
    if power > 1:
        x = torch.div(x, power)

    return x

def SNR_to_noise(snr):
    snr = 10 ** (snr / 10)
    noise_std = 1 / np.sqrt(2 * snr)

    return noise_std

# 채널 상태만 학습함
def train_mi(model, mi_net, src, snr_db, padding_idx, opt, channel):
    mi_net.train()
    opt.zero_grad()
    channels = Channels()
    # src_mask = (src == padding_idx).unsqueeze(-2).type(torch.FloatTensor).to(device)  # [batch, 1, seq_len]
    enc_output = model.encoder(src, None)
    compressed = model.time_compressor(enc_output)
    channel_enc_output = model.channel_encoder(compressed)
    Tx_sig = PowerNormalize(channel_enc_output)

    if channel == 'AWGN':
        Rx_sig = channels.AWGN(Tx_sig, snr_db)
    elif channel == 'rayleigh':
        Rx_sig = channels.Rayleigh(Tx_sig, snr_db)
    elif channel == 'rician':
        Rx_sig = channels.Rician(Tx_sig, snr_db)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

    joint, marginal = sample_batch(Tx_sig, Rx_sig)
    mi_lb, _, _ = mutual_information(joint, marginal, mi_net)
    loss_mine = -mi_lb

    loss_mine.backward()
    torch.nn.utils.clip_grad_norm_(mi_net.parameters(), 10.0)
    opt.step()

    return loss_mine.item()


# ==================== train utils ======================
# train P1 : epoch 시간 + 로스 로그 CSV 파일 저장 함수
def log_epoch_stats_csv(start_time, end_time, epoch, all_times, csv_file,
                        train_loss, val_loss, best_val_loss, lr):
    epoch_time = end_time - start_time
    all_times.append(epoch_time)
    avg_time = sum(all_times) / len(all_times)

    row = [epoch, f"{epoch_time:.2f}", f"{avg_time:.2f}",
            f"{train_loss:.6f}", f"{val_loss:.6f}", f"{best_val_loss:.6f}", f"{lr:.6f}"]

    # CSV 파일에 append
    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

    # 콘솔에도 출력
    print(f"[Epoch {epoch}] Time: {epoch_time:.2f} sec | Avg: {avg_time:.2f} sec | "
            f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | Best: {best_val_loss:.6f} | LR: {lr:.6f}")

    return all_times

# train P2 : train/val loss의 time 변화별 plot
def plot_training_logs(csv_file, save_dir):
    """
    epoch_stats.csv 파일을 읽어 학습 곡선을 플롯하고 저장하는 함수
    Best Val Loss 발생 시점을 세로선으로 표시
    """
    df = pd.read_csv(csv_file)

    # Best Val Loss 발생 Epoch 찾기
    best_idx = df["ValLoss"].idxmin()
    best_epoch = int(df.loc[best_idx, "Epoch"])
    best_val_loss = df.loc[best_idx, "ValLoss"]

    # 1. Train/Val Loss Plot
    plt.figure(figsize=(8, 6))
    plt.plot(df["Epoch"], df["TrainLoss"], label="Train Loss", marker="o")
    plt.plot(df["Epoch"], df["ValLoss"], label="Val Loss", marker="o")
    plt.axvline(x=best_epoch, color="red", linestyle="--", label=f"Best Val (Epoch {best_epoch})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Val Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=200)
    plt.close()

    # 2. Epoch Time Plot
    plt.figure(figsize=(8, 6))
    plt.plot(df["Epoch"], df["Time(sec)"], label="Epoch Time", color="orange", marker="o")
    plt.plot(df["Epoch"], df["AvgTime(sec)"], label="Avg Time", color="green", linestyle="--")
    plt.axvline(x=best_epoch, color="red", linestyle="--", label=f"Best Val (Epoch {best_epoch})")
    plt.xlabel("Epoch")
    plt.ylabel("Time (sec)")
    plt.title("Epoch Processing Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "time_curve.png"), dpi=200)
    plt.close()

    print(f"학습 로그 플롯 저장 완료: {save_dir}")
