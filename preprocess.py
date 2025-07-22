from ntpath import exists
import pandas as pd
import numpy as np
import os
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm
from torch.utils.data import TensorDataset
import joblib
import pickle
import matplotlib.pyplot as plt
import pdb
import sys
sys.path.append('C:/Users/ksshin/Desktop/ChanMinLee/time_series_deepSC_adv')
from parameters.parameters import PreprocessParams, preprocessed_data_path

"""
# preprocess.py

csv 데이터를 입력받아 정해진 cols 배열의 feature 별로 
정규화 및 sliding window를 적용 후 pt 파일로 저장하는 함수
"""
# 파라미터 가져오기
params = PreprocessParams()

folder_path=params.folder_path,
feature_cols=params.feature_cols,
batch_size=params.batch_size,
save_split_path=params.save_split_path,
split_ratio=params.split_ratio,
window_size=params.window_size,
stride=params.stride,
sample_num=params.sample_num,
PREPROCESSED_DIR=params.PREPROCESSED_DIR

# 한글 반영
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
# plt.rcParams['font.family'] = 'AppleGothic'  # Mac
# plt.rcParams['font.family'] = 'NanumGothic'  # Linux

# 마이너스 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

def is_valid_csv(fpath, expected_columns):
    try:
        df = pd.read_csv(fpath, nrows=1)
        return all(col in df.columns for col in expected_columns)
    except:
        return False

def load_all_valid_csv_tensors(folder_path, feature_cols, batch_size=8, save_split_path=None, split_ratio=0.8, window_size=128, stride=64):
    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')])
    total_files = len(files)
    valid_files = 0
    all_data = []  # 모든 원본 데이터를 먼저 수집

    # 1단계: 모든 데이터 수집 + 이상치 보정본 저장
    for fname in tqdm(files, desc="Loading CSV files"):
        fpath = os.path.join(folder_path, fname)
        if not is_valid_csv(fpath, feature_cols):
            continue
        # try:
        #     df = pd.read_csv(fpath)
        #     data = df[feature_cols].values.astype(np.float32)
        #     all_data.append(data)
        #     valid_files += 1
        try:
            df = pd.read_csv(fpath)
            # 이상치 보정
            df = df[df['Voltage_load'] < 10]
            df = df[df['Current_load'] < 2]

            os.makedirs(PREPROCESSED_DIR, exist_ok=True)
            save_path = os.path.join(PREPROCESSED_DIR, fname)
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df, columns=feature_cols)
            df.to_csv(save_path, index=False)

            data = df[feature_cols].astype(np.float32)
            # cycles = df[cycle_col].astype(np.int32)
            all_data.append(data)
            # all_cycles.append(cycles)
            valid_files += 1
        except Exception as e:
            print(f"[WARN] Failed to process {fname}: {e}")

    print(f"Valid CSV files loaded: {valid_files} / {total_files}")

    if not all_data:
        print("[ERROR] No valid data found.")
        return

    # 2단계: 전체 데이터를 하나로 합치고 스케일링
    print("Applying scaling to all data...")
    combined_data = np.vstack(all_data)  # 모든 데이터를 세로로 합침
    scaler = MinMaxScaler()
    scaled_combined = scaler.fit_transform(combined_data)

    # 3단계: 스케일링된 데이터를 다시 파일별로 분할 및 window meta 저장
    all_windows = []
    window_meta = []  # (파일명, window_start_index) 저장
    start_idx = 0
    for fname, data in zip(files, all_data):
        data_len = len(data)
        end_idx = start_idx + data_len
        scaled_data = scaled_combined[start_idx:end_idx]
        # 윈도우 생성
        for win_start in range(0, len(scaled_data) - window_size + 1, stride):
            window = scaled_data[win_start:win_start + window_size]
            all_windows.append(torch.tensor(window, dtype=torch.float32))
            window_meta.append({'file': fname, 'start': win_start})
        start_idx = end_idx

    # 텐서로 변환
    full_tensor = torch.stack(all_windows, dim=0)  # [Total_N, window, D]

    if save_split_path:
        # 저장 폴더 생성
        os.makedirs(preprocessed_data_path, exist_ok=True)

        N = full_tensor.shape[0]
        train_len = int(N * split_ratio)
        train_data = TensorDataset(full_tensor[:train_len])
        test_data = TensorDataset(full_tensor[train_len:])
        torch.save(train_data, os.path.join(save_split_path, 'train_data.pt'))
        torch.save(test_data, os.path.join(save_split_path, 'test_data.pt'))
        print(f"Saved train_data.pt ({train_len} samples), test_data.pt ({N - train_len} samples) to {save_split_path}")

        scaler_path = os.path.join(save_split_path, 'scaler.pkl')
        joblib.dump(scaler, scaler_path)
        print(f"Saved scaler to {scaler_path}")

        # window_meta도 저장
        with open(os.path.join(save_split_path, 'window_meta.pkl'), 'wb') as f:
            pickle.dump(window_meta, f)
        print(f"Saved window_meta to {os.path.join(save_split_path, 'window_meta.pkl')}")

        # 스케일링 검증
        print("\n=== 스케일링 검증 ===")
        sample_num = 1000
        sample_original = combined_data[:sample_num]  # 처음 1000개 샘플
        sample_scaled = scaled_combined[:sample_num]
        sample_restored = scaler.inverse_transform(sample_scaled)
        feature_names = feature_cols
        # verify_scaling(sample_original, sample_scaled, sample_restored, feature_names)
        plot_scaling_comparison(sample_original, sample_scaled, feature_names, sample_num)

def verify_scaling(original_data, scaled_data, restored_data, feature_names):
    for i, feature in enumerate(feature_names):
        print(f"\n{feature} 검증:")
        print(f"원본 범위: {original_data[:, i].min():.3f} ~ {original_data[:, i].max():.3f}")
        print(f"스케일링 범위: {scaled_data[:, i].min():.3f} ~ {scaled_data[:, i].max():.3f}")
        print(f"복원 범위: {restored_data[:, i].min():.3f} ~ {restored_data[:, i].max():.3f}")
        print(f"MSE: {np.mean((original_data[:, i] - restored_data[:, i])**2):.6f}")

def plot_scaling_comparison(original_data, scaled_data, feature_names, sample_num=1000):
    """
    feature별로 전처리 전/후 데이터를 한 plot에 비교해서 그려줍니다.
    sample_num: 너무 데이터가 많을 때 앞에서부터 몇 개만 그릴지 지정 (기본 1000)
    """
    n_features = len(feature_names)
    n_cols = 3
    n_rows = int(np.ceil(n_features / n_cols))
    plt.figure(figsize=(5 * n_cols, 4 * n_rows))

    for i, feature in enumerate(feature_names):
        plt.subplot(n_rows, n_cols, i + 1)
        # 데이터가 너무 많으면 앞부분만 표시
        orig = original_data[:sample_num, i]
        scaled = scaled_data[:sample_num, i]
        plt.plot(orig, label='Original', alpha=0.7)
        plt.plot(scaled, label='Scaled', alpha=0.7)
        plt.title(feature)
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.suptitle("Feature별 전처리 전/후 비교", y=1.02, fontsize=16)
    plt.show()

if __name__ == '__main__':
    load_all_valid_csv_tensors(
        folder_path=params.folder_path,
        feature_cols=params.feature_cols,
        batch_size=params.batch_size,
        save_split_path=params.save_split_path,
        split_ratio=params.split_ratio,
        window_size=params.window_size,
        stride=params.stride,
    )