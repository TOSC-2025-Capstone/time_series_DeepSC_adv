import torch
import torch.nn as nn
import joblib
import numpy as np
from tqdm import tqdm
from models.lstm import LSTMDeepSC
from models.gru import GRUDeepSC  
from models.attention_lstm import LSTMAttentionDeepSC
from models.transceiver import DeepSC
import matplotlib.pyplot as plt 
import pandas as pd
import os
import pickle
from collections import defaultdict
import pdb
from parameters.parameters import *

"""
# reconstruct_battery_series

학습 완료된 모델로 전체 배터리 시계열을 복원하는 함수 (recon.py에서 가져온 기능)
"""

train_params = TrainDeepSCParams()
test_params = TestParams()

def reconstruct_battery_series(
    model=None,
    train_pt=train_params.train_pt,
    test_pt=train_params.test_pt,
    scaler_path=train_params.scaler_path,
    window_meta_path=test_params.window_meta_path,
    device=None
    ):
    print(f"=== {model_type.upper()} 기반 전체 배터리 시계열 복원 시작 ===")
    
     # 1. 데이터 및 메타 정보 로드
    train_data = torch.load(train_pt)
    test_data = torch.load(test_pt)
    train_tensor = train_data.tensors[0]
    test_tensor = test_data.tensors[0]
    scaler = joblib.load(scaler_path)
    with open(window_meta_path, 'rb') as f:
        window_meta = pickle.load(f)
    train_len = len(train_data.tensors[0])

    # 2. 모델 로드
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = test_tensor.shape[2]
    window_size = test_tensor.shape[1]
    
    try:
        if os.path.exists(model_checkpoint_path):
            model.load_state_dict(torch.load(f'{model_checkpoint_path}best.pth', map_location=device))
        model.eval()
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        return

    # 3. 배터리별로 시계열 길이 추정 (원본 csv 불러와서 추정)
    battery_files = sorted(set([meta['file'] for meta in window_meta]))
    battery_lengths = {}
    for fname in battery_files:
        df = pd.read_csv(os.path.join(outlier_cut_csv_path, fname))
        battery_lengths[fname] = len(df)

    # 4. 배터리별로 빈 시계열 배열 준비 (정규화된 값으로)
    reconstructed = {fname: np.zeros((battery_lengths[fname], input_dim)) for fname in battery_files}
    counts = {fname: np.zeros(battery_lengths[fname]) for fname in battery_files}

    # 5. 각 window 복원 및 배터리별 시계열에 합치기 (정규화된 값으로)
    with torch.no_grad():
        for i in tqdm(range(test_tensor.shape[0]), desc="Reconstructing"):
            input_data = test_tensor[i].unsqueeze(0).to(device)
            output = model(input_data)
            output_np = output.squeeze(0).cpu().numpy()  # (window, feature)

            meta = window_meta[train_len + i]  # test set은 train 다음부터 시작
            fname = meta['file']
            start = meta['start']
            actual_len = min(window_size, battery_lengths[fname] - start)
            # 정규화된 값으로 합치기
            reconstructed[fname][start:start+actual_len] += output_np[:actual_len]
            counts[fname][start:start+actual_len] += 1

    # 6. 겹치는 부분 평균내기 (정규화된 값)
    for fname in battery_files:
        mask = counts[fname] > 0
        reconstructed[fname][mask] /= counts[fname][mask][:, None]

    # 7. 평균낸 후 역정규화 및 저장/시각화
    os.makedirs(save_recon_dir, exist_ok=True)
    os.makedirs(save_fig_dir, exist_ok=True)
    for fname in battery_files:
        base = os.path.splitext(fname)[0]

        # === 여기서 역정규화 ===
        recon_inv = scaler.inverse_transform(reconstructed[fname])
        df_recon = pd.DataFrame(recon_inv, columns=feature_cols)
        csv_path = os.path.join(save_recon_dir, f'{base}_reconstructed.csv')
        df_recon.to_csv(csv_path, index=False)
        print(f"복원된 전체 시계열 저장: {csv_path}")

        # 비교 시각화 (test set에 포함된 배터리만)
        if np.any(counts[fname] > 0):
            df_orig = pd.read_csv(os.path.join(outlier_cut_csv_path, fname))
            plt.figure(figsize=(15, 10))
            for i, col in enumerate(feature_cols):
                plt.subplot(2, 3, i+1)
                plt.plot(df_orig[col], label='Original', alpha=0.7)
                plt.plot(df_recon[col], label=f'{model_type.upper()} Reconstructed', alpha=0.7)
                plt.title(col)
                plt.legend()
                plt.grid(True)
            plt.suptitle(f'{model_type.upper()} Comparison: {fname}')
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            fig_path = os.path.join(save_fig_dir, f'{base}_compare.png')
            plt.savefig(fig_path, dpi=200)
            plt.close()
            print(f"비교 그래프 저장: {fig_path}")
            
            # Residual(오차) 시계열 플롯 추가
            plt.figure(figsize=(15, 10))
            for i, col in enumerate(feature_cols):
                plt.subplot(2, 3, i+1)
                residual = df_orig[col] - df_recon[col]
                plt.plot(residual, label='Residual', color='orange', alpha=0.8)
                plt.title(f'Residual: {col}')
                plt.axhline(0, color='gray', linestyle='--', linewidth=1)
                plt.legend()
                plt.grid(True)
            plt.suptitle(f'{model_type.upper()} Residuals: {fname}')
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            residual_fig_path = os.path.join(save_fig_dir, f'{base}_residual.png')
            plt.savefig(residual_fig_path, dpi=200)
            plt.close()
            print(f"Residual 그래프 저장: {residual_fig_path}")

            # === 복원 오차율(%) 플롯 추가 ===
            plt.figure(figsize=(15, 10))
            epsilon = 1e-9
            for i, col in enumerate(feature_cols):
                plt.subplot(2, 3, i+1)
                # 오차율(%) 계산
                residual_percent = np.abs(df_orig[col] - df_recon[col]) / (np.abs(df_orig[col]) + epsilon) * 100
                # pdb.set_trace()
                plt.plot(residual_percent, label='Residual %', color='orange', alpha=0.8)
                plt.title(f'Residual %: {col}')
                plt.ylabel('Residual (%)')
                plt.axhline(0, color='gray', linestyle='--', linewidth=1)
                plt.legend()
                plt.grid(True)
            plt.suptitle(f'{model_type.upper()} Residual Percent: {fname}')
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            residual_percent_fig_path = os.path.join(save_fig_dir, f'{base}_residual_percent.png')
            plt.savefig(residual_percent_fig_path, dpi=200)
            plt.close()
            print(f"Residual Percent 그래프 저장: {residual_percent_fig_path}")

    # 8. 카운트 배열 확인
    for fname in battery_files:
        print(f"{fname} counts unique: {np.unique(counts[fname])}")

# if __name__ == "__main__":
