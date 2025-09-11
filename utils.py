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

class Channels():

    def AWGN(self, Tx_sig, n_var=0.1):
        Rx_sig = Tx_sig + torch.normal(0, 0.2, size=Tx_sig.shape).to(device)

        # 0909 노이즈 추가 비교
        # Rx_sig_copy = Rx_sig
        # Tx_sig_copy = Tx_sig

        # flatten = nn.Flatten()
        # Rx_sig_flatten = flatten(Rx_sig_copy)
        # Tx_sig_flatten = flatten(Tx_sig_copy)
        # Rx_sig_flatten = Rx_sig_flatten.view(1, -1)
        # Tx_sig_flatten = Tx_sig_flatten.view(1, -1)

        # Rx_sig_df = pd.DataFrame(Rx_sig_flatten.cpu().detach().numpy())
        # Tx_sig_df = pd.DataFrame(Tx_sig_flatten.cpu().detach().numpy())

        # transposed_Rx_sig_df = Rx_sig_df.transpose()
        # transposed_Tx_sig_df = Tx_sig_df.transpose()

        # plt.figure(figsize=(15, 10))
        # plt.hist(
        #     transposed_Rx_sig_df,
        #     bins=50,
        #     alpha=0.5,
        #     label="Rx",
        #     density=True,
        #     color="#3498db",
        # )
        # plt.hist(
        #     transposed_Tx_sig_df,
        #     bins=50,
        #     alpha=0.8,
        #     label="Tx",
        #     density=True,
        #     color="#e74c3c"
        # )
        # plt.show()
        return Rx_sig

    def Rayleigh(self, Tx_sig, n_var=0.1):
        shape = Tx_sig.shape
        H_real = torch.normal(0, math.sqrt(1/2), size=[1]).to(device)
        H_imag = torch.normal(0, math.sqrt(1/2), size=[1]).to(device)
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(device)
        Tx_sig = torch.matmul(Tx_sig.view(shape[0], -1, 2), H)
        Rx_sig = self.AWGN(Tx_sig, n_var)
        # Channel estimation
        Rx_sig = torch.matmul(Rx_sig, torch.inverse(H)).view(shape)

        return Rx_sig

    def Rician(self, Tx_sig, n_var=0.1, K=1):
        shape = Tx_sig.shape
        mean = math.sqrt(K / (K + 1))
        std = math.sqrt(1 / (K + 1))
        H_real = torch.normal(mean, std, size=[1]).to(device)
        H_imag = torch.normal(mean, std, size=[1]).to(device)
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(device)
        Tx_sig = torch.matmul(Tx_sig.view(shape[0], -1, 2), H)
        Rx_sig = self.AWGN(Tx_sig, n_var)
        # Channel estimation
        Rx_sig = torch.matmul(Rx_sig, torch.inverse(H)).view(shape)

        return Rx_sig

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
def train_mi(model, mi_net, src, n_var, padding_idx, opt, channel):
    mi_net.train()
    opt.zero_grad()
    channels = Channels()
    # src_mask = (src == padding_idx).unsqueeze(-2).type(torch.FloatTensor).to(device)  # [batch, 1, seq_len]
    enc_output = model.encoder(src, None)
    compressed = model.time_compressor(enc_output)
    channel_enc_output = model.channel_encoder(compressed)
    Tx_sig = PowerNormalize(channel_enc_output)

    if channel == 'AWGN':
        Rx_sig = channels.AWGN(Tx_sig, n_var)
    elif channel == 'rayleigh':
        Rx_sig = channels.Rayleigh(Tx_sig, n_var)
    elif channel == 'rician':
        Rx_sig = channels.Rician(Tx_sig, n_var)
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
