import os
import pandas as pd
import numpy as np
from scipy import interpolate
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
from torch.utils.data import TensorDataset
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import pdb

# 한글깨짐 방지
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

"""
# methods.py
P1,2,3 에 필요한 서브 기능 함수들을 모듈화함 

# 처리 순서
P1. outlier_eliminate.py 
original dataset에서 배터리 메타데이터를 로드하고 특정 배터리 데이터를 필터링
discharge 파일 목록을 생성하고, 각 파일을 읽어서 통합된 데이터프레임을 생성
이상치 탐지 및 제거 후 통합된 데이터프레임을 반환

P2. prepare_data.py
1의 통합된 데이터프레임을 사용하여 피쳐 별 스케일 정규화를 진행
정규화를 마친 통합 데이터프레임을 파일 단위로 분리 후
cycle_reshape.py의 함수를 사용하여 각 파일을 256개 샘플로 리샘플링
각 샘플들을 텐서로 변환하여 학습/테스트 데이터셋을 생성
각 파일을 pickle, pt로 저장

P3. (optional) preprocessed_data_check.py
preprocessed Tensor 비교 플랏
"""


# P1.1
def load_and_filter_metadata(exclude_batteries=None):
    """
    메타데이터를 로드하고 특정 배터리 데이터를 필터링

    Args:
        exclude_batteries (list): 제외할 배터리 ID 리스트

    Returns:
        pd.DataFrame: 필터링된 discharge 데이터
        list: 처리할 파일 목록
    """
    meta = pd.read_csv("original_dataset/metadata.csv")
    discharge_data = meta[meta["type"] == "discharge"]

    print(f"\n제거 전 discharge 데이터 개수: {len(discharge_data)}")
    print(
        "제거 전 포함된 배터리:",
        sorted(list(discharge_data["battery_id"].str.extract(r"(B\d{4})")[0].unique())),
    )

    if exclude_batteries:
        condition = discharge_data["battery_id"].str.contains(
            "|".join(exclude_batteries)
        )
        discharge_data = discharge_data[~condition]

    print(f"\n제거 후 discharge 데이터 개수: {len(discharge_data)}")
    print(
        "제거 후 포함된 배터리:",
        sorted(list(discharge_data["battery_id"].str.extract(r"(B\d{4})")[0].unique())),
    )

    discharge_files = sorted(
        [f"{int(fname.split('.')[0]):05d}.csv" for fname in discharge_data["filename"]]
    )

    return discharge_data, discharge_files


# P1.2
def load_csv_data(discharge_files, input_folder):
    """
    리샘플링된 데이터 파일들을 로드하여 통합

    Args:
        discharge_files (list): 처리할 파일 목록
        input_folder (str): 리샘플링된 데이터가 있는 폴더 경로

    Returns:
        pd.DataFrame: 통합된 데이터프레임
    """
    all_data = []
    missing_files = []

    print("\n데이터 파일 확인 중...")
    for fname in discharge_files:
        fpath = os.path.join(input_folder, fname)
        if not os.path.exists(fpath):
            missing_files.append(fname)

    if missing_files:
        print(f"\n경고: {len(missing_files)}개의 파일이 {input_folder}에 없습니다.")
        print("먼저 cycle_reshape.py를 실행하여 resampled 데이터를 생성해주세요.")
        print("첫 번째 없는 파일:", missing_files[0])
        return None

    # discharge_files에 있는 파일이름만 가져와서 합치기
    for fname in tqdm(discharge_files, desc="파일 로딩"):
        fpath = os.path.join(input_folder, fname)
        try:
            df = pd.read_csv(fpath)
            if len(df) > 0:
                # 파일명에서 인덱스 추출 (00001.csv -> 1)
                file_index = int(fname.split(".")[0])
                # file_index 컬럼 추가
                df["file_index"] = file_index
                all_data.append(df)
            else:
                print(f"\n경고: {fname}가 비어있습니다.")
        except Exception as e:
            print(f"\n오류: {fname} 로딩 중 문제 발생 - {str(e)}")

    if not all_data:
        print("\n오류: 로드된 데이터가 없습니다!")
        return None

    total_df = pd.concat(all_data, ignore_index=True)
    print(
        f"통합 데이터 크기: {total_df.shape} (파일 {len(all_data)}개 성공적으로 로드)"
    )
    return total_df


# P1.3
def visualize_data_comparison(total_df, cleaned_df, feature_names, output_dir):
    """
    원본 데이터와 처리된 데이터의 비교 시각화(히스토그램, 선 그래프, 산점도)

    두 가지 비교 플롯 생성:
    분포 비교 (히스토그램): outlier_removal_comparison.png
    각 특성별로 원본 데이터와 처리된 데이터의 분포 비교
    시계열 비교: outlier_removal_timeseries.png
    첫 번째 파일(256개 샘플)에 대해 원본과 처리된 시계열 비교
    각 특성별로 시간에 따른 변화 비교 가능

    Args:
        total_df (pd.DataFrame): 원본 데이터
        cleaned_df (pd.DataFrame): 처리된 데이터
        feature_names (list): 특성 이름 리스트
        output_dir (str): 출력 디렉토리 경로
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. 히스토그램 비교
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(total_df.columns):
        if col != "cycle_idx":
            plt.subplot(2, 3, i + 1)
            plt.hist(
                total_df[col],
                bins=50,
                alpha=0.8,
                label="원본 데이터",
                density=True,
                color="#3498db",
            )
            plt.hist(
                cleaned_df[col],
                bins=50,
                alpha=0.5,
                label="이상치 제거 후",
                density=True,
                color="#e74c3c",
            )
            plt.title(f"{feature_names[i]} Distribution")
            plt.xlabel("Value")
            plt.ylabel("Density")
            plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "distribution_comparison.png"))
    plt.close()

    # 2. 선 그래프 비교 (첫 번째 파일)
    plt.figure(figsize=(15, 10))
    first_file_data = total_df.iloc[:256]
    first_file_cleaned = cleaned_df.iloc[:256]

    for i, col in enumerate(total_df.columns):
        if col != "cycle_idx":
            plt.subplot(2, 3, i + 1)
            plt.plot(
                first_file_data[col],
                alpha=0.8,
                label="원본 데이터",
                color="#3498db",
                linewidth=2,
            )
            plt.plot(
                first_file_cleaned[col],
                alpha=0.5,
                label="이상치 제거 후",
                color="#e74c3c",
                linewidth=2,
            )
            plt.title(f"{feature_names[i]} Time Series")
            plt.xlabel("Time Step")
            plt.ylabel("Value")
            plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "timeseries_comparison.png"))
    plt.close()

    # 3. 산점도 비교
    plt.figure(figsize=(15, 10))
    sample_size = min(1000, len(total_df))
    for i, col in enumerate(total_df.columns):
        if col != "cycle_idx":
            plt.subplot(2, 3, i + 1)
            plt.scatter(
                range(sample_size),
                total_df[col].iloc[:sample_size],
                alpha=0.8,
                label="원본 데이터",
                color="#3498db",
                s=20,
            )
            plt.scatter(
                range(sample_size),
                cleaned_df[col].iloc[:sample_size],
                alpha=0.5,
                label="이상치 제거 후",
                color="#e74c3c",
                s=20,
            )
            plt.title(f"{feature_names[i]} Scatter Plot")
            plt.xlabel("Data Point")
            plt.ylabel("Value")
            plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "scatter_comparison.png"))
    plt.close()


# P2.1
def scale_data(df, scaler_type="minmax"):
    """
    데이터 스케일링 (MinMax 또는 Z-score)
    """
    if scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:  # 'zscore'
        scaler = StandardScaler()

    # 스케일링 적용
    scaled_data = scaler.fit_transform(df)

    # DataFrame 재구성 (원래 컬럼명 유지)
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns)

    return scaled_df, scaler


# P2.2
def grouping_df(df):
    """
    file_index로 그룹화하여 각 파일별 데이터프레임 생성
    """
    # file_index 컬럼이 있는지 확인
    if "file_index" not in df.columns:
        raise ValueError("DataFrame에 'file_index' 컬럼이 없습니다.")

    # file_index로 그룹화하여 파일별 데이터프레임 생성
    grouped_dfs = {}
    for file_index, group_df in df.groupby("file_index"):
        # file_index 컬럼 제외하고 데이터만 저장
        grouped_dfs[file_index] = group_df.drop("file_index", axis=1)

    print(f"\n분리된 파일 수: {len(grouped_dfs)}")

    # 각 파일의 행 수 통계
    file_sizes = [df.shape[0] for df in grouped_dfs.values()]
    print(
        f"파일당 행 수 - 최소: {min(file_sizes)}, 최대: {max(file_sizes)}, 평균: {np.mean(file_sizes):.2f}"
    )
    return grouped_dfs


# P2.4
def split_and_transform_data(scaled_df, discharge_files, test_ratio=0.2):
    """
    데이터를 학습/테스트 세트로 분할하고 텐서로 변환

    Args:
        scaled_df (pd.DataFrame): 스케일링된 데이터
        discharge_files (list): 전체 파일 목록 (파일명 format: 00001.csv)
        test_ratio (float): 테스트 세트 비율

    Returns:
        torch.Tensor: 학습 데이터
        torch.Tensor: 테스트 데이터
        list: 학습 세트 파일 인덱스
        list: 테스트 세트 파일 인덱스
    """
    pdb.set_trace()  # 디버깅용
    n_files = len(discharge_files)
    n_samples_per_file = 256
    n_features = len(scaled_df.columns)

    # 파일 인덱스 추출 (discharge_files는 00001 부터 담긴 리스트)
    file_indices = discharge_files

    # 테스트 세트 파일 인덱스 선택
    # seed 42에 따라 동일한 테스트 세트가 생성되도록 설정 (pesudo-random)
    np.random.seed(42)
    test_file_idx = np.random.choice(
        np.arange(n_files), size=int(n_files * test_ratio), replace=False
    )
    test_file_indices = [file_indices[i] for i in test_file_idx]
    train_file_indices = [idx for idx in file_indices if idx not in test_file_indices]

    # 데이터 인덱스로 변환
    test_data_indices = []
    for idx in test_file_idx:
        start_idx = idx * n_samples_per_file
        test_data_indices.extend(range(start_idx, start_idx + n_samples_per_file))

    # 마스크 생성
    is_test = np.zeros(len(scaled_df), dtype=bool)
    is_test[test_data_indices] = True

    # 데이터 변환
    train_samples = scaled_df[~is_test].values
    test_samples = scaled_df[is_test].values

    n_train_files = len(train_samples) // n_samples_per_file
    n_test_files = len(test_samples) // n_samples_per_file

    # 텐서로 변환
    train_data = torch.FloatTensor(train_samples).view(
        n_train_files, n_samples_per_file, n_features
    )
    test_data = torch.FloatTensor(test_samples).view(
        n_test_files, n_samples_per_file, n_features
    )

    return train_data, test_data, sorted(train_file_indices), sorted(test_file_indices)


# P2.5
def save_tensor_dataset(train_data, test_data, scaler, output_folder):
    print(f"\n데이터 shape 확인:")
    print(f"train_data: {train_data.shape} (파일 수 x 256 x 특성 수)")
    print(f"test_data: {test_data.shape} (파일 수 x 256 x 특성 수)")

    # scaler가 minmax인지 zscore인지 확인
    scaler_type = "minmax" if isinstance(scaler, MinMaxScaler) else "zscore"
    print(f"사용된 스케일러 타입: {scaler_type}")

    # 첫 번째 파일의 형태 확인
    print(f"\n첫 번째 학습 파일의 shape: {train_data[0].shape}")  # 256 x 6 이어야 함
    print("첫 번째 학습 파일의 처음 3개 행:")
    print(train_data[0][:3])  # 처음 3개 행의 값을 출력

    # 6. TensorDataset 생성 및 결과 저장
    output_folder = output_folder + f"processed_{scaler_type}/"
    os.makedirs(output_folder, exist_ok=True)

    # TensorDataset 생성 (향후 추가 레이블이나 메타데이터를 위해 확장 가능)
    train_dataset = TensorDataset(train_data)
    test_dataset = TensorDataset(test_data)

    # 데이터셋 저장 -> 6 2 2
    torch.save(train_dataset, os.path.join(output_folder, "train_data.pt"))
    torch.save(test_dataset, os.path.join(output_folder, "test_data.pt"))

    # 스케일러 저장
    with open(os.path.join(output_folder, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    # 처리된 데이터 통계 출력
    print("\n=== 처리 완료 ===")
    print(f"Train 데이터: {train_data.shape}")
    print(f"Test 데이터: {test_data.shape}")
    print(f"결과가 {output_folder}에 저장되었습니다.")
