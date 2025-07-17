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
        Rx_sig = Tx_sig + torch.normal(0, n_var, size=Tx_sig.shape).to(device)
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

# 수정 필요 07-17
def compare_original_reconstructed(model_type="deepsc"):
    """
    원본 데이터와 복원 데이터를 비교하는 함수
    """
    print(f"=== {model_type.upper()} 원본 vs 복원 데이터 비교 ===")
    
    # 1. 원본 데이터 로드
    df_orig = pd.read_csv('data_handling/merged/B0005.csv')
    feature_cols = ['Voltage_measured', 'Current_measured', 'Temperature_measured', 'Current_load', 'Voltage_load', 'Time']

    # 2. 복원 데이터 로드 (예: 첫 window)
    save_dir = f'reconstructed_{model_type}'
    df_recon = pd.read_csv(f'{save_dir}/B0005_window0.csv')

    # 3. window의 시작 인덱스(예: 0)와 window_size(예: 128) 지정
    window_start = 0
    window_size = df_recon.shape[0]
    df_orig_window = df_orig[feature_cols].iloc[window_start:window_start+window_size].reset_index(drop=True)

    # 4. 비교 시각화
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(feature_cols):
        plt.subplot(2, 3, i+1)
        plt.plot(df_orig_window[col], label='Original')
        plt.plot(df_recon[col], label=f'{model_type.upper()} Reconstructed')
        plt.title(col)
        plt.legend()
        plt.grid(True)
    plt.suptitle(f'{model_type.upper()} vs Original Comparison')
    plt.tight_layout()
    plt.show()


# 수정 필요 07-17
def compare_all_models():
    """
    모든 모델의 성능을 비교하는 함수
    """
    print("=== 모든 모델 성능 비교 ===")
    
    models = ["deepsc", "lstm", "gru"]
    results = {}
    
    for model_type in models:
        print(f"\n{model_type.upper()} 모델 테스트 중...")
        try:
            # 간단한 테스트를 위해 1개 샘플만 사용
            test_data = torch.load('model/preprocessed_data/test_data.pt')
            test_tensor = test_data.tensors[0]
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            input_dim = test_tensor.shape[2]
            window_size = test_tensor.shape[1]
            
            model, checkpoint_path = create_model(model_type, input_dim, window_size, device)
            if os.path.exists(checkpoint_path):
                model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            model.eval()
            
            # 테스트
            with torch.no_grad():
                input_data = test_tensor[0].unsqueeze(0).to(device)
                output = model(input_data)
                mse = nn.MSELoss()(output, input_data).item()
                
                if model_type == "deepsc":
                    encoded = model.encoder(input_data, src_mask=None)
                    channel_encoded = model.channel_encoder(encoded)
                    compressed_size = channel_encoded.numel()
                else:
                    compression_ratio = model.get_compression_ratio()
                    compressed_size = int(input_data.numel() * compression_ratio)
                
                original_size = input_data.numel()
                compression_ratio = compressed_size / original_size
                
                results[model_type] = {
                    'mse': mse,
                    'compression_ratio': compression_ratio,
                    'compression_efficiency': (1 - compression_ratio) * 100
                }
                
                print(f"{model_type.upper()}: MSE={mse:.6f}, 압축률={compression_ratio:.3f}, 효율성={(1-compression_ratio)*100:.1f}%")
                
        except Exception as e:
            print(f"{model_type.upper()} 모델 테스트 실패: {e}")
            results[model_type] = {'error': str(e)}
    
    # 결과 요약
    print(f"\n=== 모델 비교 결과 요약 ===")
    for model_type, result in results.items():
        if 'error' not in result:
            print(f"{model_type.upper()}:")
            print(f"  MSE: {result['mse']:.6f}")
            print(f"  압축률: {result['compression_ratio']:.3f}")
            print(f"  압축 효율성: {result['compression_efficiency']:.1f}%")
        else:
            print(f"{model_type.upper()}: 오류 - {result['error']}")