import os
import pdb
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import interpolate
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import TensorDataset
from tqdm import tqdm

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
        # 여기 (exclude batteries 반전)
        discharge_data = discharge_data[~condition]
        # discharge_data = discharge_data[condition]

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
    원본 데이터와 처리된 데이터의 비교 시각화(히스토그램, 선 그래프, 산점도, 시계열 샘플, 파일별 데이터 분포)
    다섯 가지 비교 플롯 생성:
    1. 분포 비교 (히스토그램): distribution_comparison.png
        각 특성별로 원본 데이터와 처리된 데이터의 분포 비교
    2. 시계열 비교: timeseries_comparison.png
        첫 번째 파일(256개 샘플)에 대해 원본과 처리된 시계열 비교
    3. 산점도 비교: scatter_comparison.png
        각 특성별로 시간에 따른 변화를 산점도로 비교
    4. 시계열 샘플 분석: timeseries_sample_analysis.png
        Current_measured의 시간 순서 샘플링 분석 및 파일별 데이터 분포
    5. 추가적인 시각화가 필요한 경우 확장 가능

    Args:
        total_df (pd.DataFrame): 원본 데이터
        cleaned_df (pd.DataFrame): 처리된 데이터
        feature_names (list): 특성 이름 리스트
        output_dir (str): 출력 디렉토리 경로
    """
    import os

    import matplotlib.pyplot as plt

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

    # 4. 시계열 샘플 분석 및 파일별 데이터 분포 (원본 vs 처리된 데이터 비교)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Current_measured Analysis: Original vs Cleaned Data", fontsize=16)

    # 4-1. 원본 데이터 시계열 샘플 플롯 (왼쪽 상단)
    tdata_original = total_df.sample(len(total_df))
    axes[0, 0].scatter(
        range(len(tdata_original)),
        tdata_original["Current_measured"],
        alpha=0.6,
        s=1,
        color="red",
    )
    axes[0, 0].set_title("Original Data - Time Series Sample Plot")
    axes[0, 0].set_xlabel("Index")
    axes[0, 0].set_ylabel("Current_measured")
    axes[0, 0].grid(True, alpha=0.3)

    # 4-2. 처리된 데이터 시계열 샘플 플롯 (오른쪽 상단)
    tdata_cleaned = cleaned_df.sample(len(cleaned_df))
    axes[0, 1].scatter(
        range(len(tdata_cleaned)),
        tdata_cleaned["Current_measured"],
        alpha=0.6,
        s=1,
        color="blue",
    )
    axes[0, 1].set_title("Cleaned Data - Time Series Sample Plot")
    axes[0, 1].set_xlabel("Index")
    axes[0, 1].set_ylabel("Current_measured")
    axes[0, 1].grid(True, alpha=0.3)

    # 4-3. 원본 데이터 시계열 샘플 플롯 (왼쪽 하단)
    tdata_original = total_df.sample(len(total_df))
    axes[1, 0].scatter(
        range(len(tdata_original)),
        tdata_original["Current_load"],
        alpha=0.6,
        s=1,
        color="red",
    )
    axes[1, 0].set_title("Original Data - Time Series Sample Plot")
    axes[1, 0].set_xlabel("Index")
    axes[1, 0].set_ylabel("Current_load")
    axes[1, 0].grid(True, alpha=0.3)

    # 4-4. 처리된 데이터 시계열 샘플 플롯 (오른쪽 하단)
    tdata_cleaned = cleaned_df.sample(len(cleaned_df))
    axes[1, 1].scatter(
        range(len(tdata_cleaned)),
        tdata_cleaned["Current_load"],
        alpha=0.6,
        s=1,
        color="blue",
    )
    axes[1, 1].set_title("Cleaned Data - Time Series Sample Plot")
    axes[1, 1].set_xlabel("Index")
    axes[1, 1].set_ylabel("Current_load")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "timeseries_sample_analysis.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"모든 시각화 파일이 {output_dir} 디렉토리에 저장되었습니다.")
    print("생성된 파일들:")
    print("- distribution_comparison.png: 분포 비교")
    print("- timeseries_comparison.png: 시계열 비교")
    print("- scatter_comparison.png: 산점도 비교")
    print("- timeseries_sample_analysis.png: 시계열 샘플 분석 및 파일별 분포")


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
        # grouped_dfs[file_index] = group_df.drop("file_index", axis=1)
        grouped_dfs[file_index] = group_df

    print(f"\n분리된 파일 수: {len(grouped_dfs)}")

    # 각 파일의 행 수 통계
    file_sizes = [df.shape[0] for df in grouped_dfs.values()]
    print(
        f"파일당 행 수 - 최소: {min(file_sizes)}, 최대: {max(file_sizes)}, 평균: {np.mean(file_sizes):.2f}"
    )
    return grouped_dfs


# P2.4
def split_and_transform_data(
    scaled_df, discharge_files, grouped_total_df_indices, val_ratio=0.2, test_ratio=0.2, segment_target_len=None
):
    """
    데이터를 학습/검증/테스트 세트로 분할하고 텐서로 변환 (6:2:2 비율)

    Args:
        scaled_df (pd.DataFrame): 스케일링된 데이터
        discharge_files (list): 전체 파일 목록 (파일명 format: 00001.csv)
        val_ratio (float): 검증 세트 비율
        test_ratio (float): 테스트 세트 비율

    Returns:
        torch.Tensor: 학습 데이터
        torch.Tensor: 검증 데이터
        torch.Tensor: 테스트 데이터
        list: 학습 세트 파일 인덱스
        list: 검증 세트 파일 인덱스
        list: 테스트 세트 파일 인덱스
    """

    # scaled_df 의 파일 인덱스, all_indices, ~~_file_index는 2694개 에서 0~2693의 서수 개념
    # file_indices는 실제 파일명의 숫자만 가져온 리스트 1,5,7,9 ... 7564

    n_files = len(discharge_files)
    # segment_target_len = 256 -> 가변으로 변함
    n_features = len(scaled_df.columns)

    # 파일 인덱스 추출 (discharge_files는 00001 부터 담긴 리스트)
    file_indices = discharge_files # 1 3 5 ... 7564
    all_indices = np.arange(n_files) # 0 ~ 2693

    # 테스트 세트 선택
    n_test = int(n_files * test_ratio)
    test_file_idx = np.random.choice(all_indices, size=n_test, replace=False)
    remaining_idx = np.array([i for i in all_indices if i not in test_file_idx])

    # 검증 세트 선택
    n_val = int(n_files * val_ratio)
    val_file_idx = np.random.choice(remaining_idx, size=n_val, replace=False)

    # 여기 (학습 비율 train 100%)
    train_file_idx = np.array([i for i in remaining_idx if i not in val_file_idx])
    # train_file_idx = all_indices

    # 각 세트의 파일 인덱스 추출
    test_file_indices = [file_indices[i] for i in test_file_idx]
    val_file_indices = [file_indices[i] for i in val_file_idx]
    train_file_indices = [file_indices[i] for i in train_file_idx]

    # 데이터 인덱스로 변환
    def create_data_indices(dataset_file_idx):
        indices = []

        # 'file_index'를 기준으로 DataFrame을 그룹화합니다.
        # sort=False는 DataFrame의 원래 순서를 유지합니다.
        # groups는 딕셔너리를 반환합니다.
        # 예: { 1: [0, 1, 2, ...],  2: [500, 501, ...], ... , 7564: [..., 745602] }
        #    여기서 [0, 1, 2...]는 scaled_df의 '병합된' 원본 인덱스입니다.
        group_indices_dict = scaled_df.groupby('file_index', sort=False).groups

        # 딕셔너리를 순회합니다.
        for idx in dataset_file_idx:
            # 딕셔너리에서 'idx' 키로 값을 직접 찾습니다. idx는 원래 파일명 인덱스임 (1~2694 아니고 1,3,5 ... 7564까지 있는거)
            if idx in group_indices_dict:
                # original_indices_list는 [500, 501, 502, ...] 같은
                # '원본 인덱스 리스트'입니다.
                original_indices_list = group_indices_dict[idx]

                # 이 리스트를 'indices'에 통째로 추가합니다.
                indices.extend(original_indices_list)
            else:
                print(f"경고: file_index {idx}를 scaled_df에서 찾을 수 없습니다.")
        return indices

    # test_data_indices = create_data_indices(test_file_idx)
    # val_data_indices = create_data_indices(val_file_idx)
    # train_data_indices = create_data_indices(train_file_idx)
    test_data_indices = create_data_indices(test_file_indices)
    val_data_indices = create_data_indices(val_file_indices)
    train_data_indices = create_data_indices(train_file_indices)

    # 마스크 생성 및 데이터 분할
    def create_mask_and_transform(indices, total_length):
        mask = np.zeros(total_length, dtype=bool)
        mask[indices] = True
        return mask

    is_test = create_mask_and_transform(test_data_indices, len(scaled_df))
    is_val = create_mask_and_transform(val_data_indices, len(scaled_df))
    is_train = create_mask_and_transform(train_data_indices, len(scaled_df))

    # 데이터 변환
    train_samples = scaled_df[is_train].values
    val_samples = scaled_df[is_val].values
    test_samples = scaled_df[is_test].values

    # 텐서로 변환
    train_data = torch.FloatTensor(train_samples).view(-1, segment_target_len, n_features)
    val_data = torch.FloatTensor(val_samples).view(-1, segment_target_len, n_features)
    test_data = torch.FloatTensor(test_samples).view(-1, segment_target_len, n_features)

    # final_train_data = add_segment_index(train_data)
    # final_val_data = add_segment_index(val_data)
    # final_test_data = add_segment_index(test_data)

    return (
        train_data,
        val_data,
        test_data,
        sorted(train_file_indices),
        sorted(val_file_indices),
        sorted(test_file_indices),
    )

# P2.4.1
def add_cycle_index_for_battery_id(tensor_data):
    print()

# def add_segment_index(tensor_data):
#     # N, L, C_in = 55718, 8, 7
#     N = len(tensor_data)
#     L = len(tensor_data[0])
#     C_in = len(tensor_data[0][0])

#     # 1. 각 세그먼트의 'file_index'를 대표하는 1D 텐서를 추출합니다.
#     # (세그먼트 내 8개 행의 file_index는 모두 동일하므로, 0번째 행의 값만 사용)
#     # file_indices shape: [55718]
#     file_indices = tensor_data[:, 0, -1]

#     # 2. 'file_index'가 변경되는 직전 지점을 찾습니다. (True/False 마스크)
#     # (예: [5, 5, 7, 7, 9] -> [False, True, False, True])
#     # key_changes shape: [55717]
#     key_changes = file_indices[1:] != file_indices[:-1]

#     # 3. 'is_new_group_mask' 마스크를 생성합니다.
#     # 맨 첫 번째 세그먼트(True)와, file_index가 변경되는 지점(True)을 표시합니다.
#     # (예: [True, False, True, False, True])
#     # is_new_group_mask shape: [55718]
#     is_new_group_mask = torch.cat(
#         (torch.tensor([True], device=tensor_data.device), key_changes)
#     )

#     # 4. 'is_new_group_mask'가 True인 위치(인덱스)를 찾습니다.
#     # 이는 각 그룹(동일한 file_index)이 시작되는 인덱스 리스트입니다.
#     # (예: [0, 2, 4])
#     # start_indices shape: [Number of groups]
#     start_indices = is_new_group_mask.nonzero().flatten()

#     # 5. 0부터 N-1까지의 전체 세그먼트 인덱스를 생성합니다.
#     # full_range shape: [55718]
#     full_range = torch.arange(N, device=tensor_data.device)

#     # 6. 각 세그먼트(full_range)가 어떤 그룹(start_indices)에 속하는지 찾습니다.
#     # torch.searchsorted는 'full_range'의 각 요소를 'start_indices'의 어디에
#     # 삽입해야 정렬이 유지되는지 알려줍니다.
#     # (예: group_indices [0, 0, 1, 1, 2])
#     group_indices = torch.searchsorted(start_indices, full_range, right=True) - 1

#     # 7. 각 세그먼트가 속한 그룹(사이클 파일)의 '세그먼트의 첫 행 인덱스'를 가져옵니다.
#     # (예: start_indices_per_segment [0, 0, 2, 2, 4])
#     start_indices_per_segment = start_indices[group_indices]

#     # 8. (핵심) 전체 인덱스에서 그룹 시작 인덱스를 빼서 그룹 내 순서(0-based)를 계산합니다.
#     # full_range:              [0, 1, 2, 3, 4]
#     # start_indices_per_segment: [0, 0, 2, 2, 4]
#     # 빼기 결과 (0-based):    [0, 1, 0, 1, 0]
#     sequence_idx_zero_based = full_range - start_indices_per_segment

#     # 9. 1을 더해 1-based 순서 인덱스로 만듭니다. (요청사항 반영)
#     # (결과: [1, 2, 1, 2, 1])
#     # sequence_idx_one_based shape: [55718]
#     sequence_idx_one_based = (sequence_idx_zero_based + 1).to(tensor_data.dtype)

#     # 10. 이 1D 텐서를 [N, L, 1] shape으로 확장(expand)합니다.
#     # [55718] -> [55718, 1] -> [55718, 1, 1]
#     new_feature_tensor = sequence_idx_one_based.unsqueeze(1).unsqueeze(2)
#     # [55718, 1, 1] -> [55718, 8, 1] (8개 행 모두에 동일한 순서 인덱스 적용)
#     new_feature_tensor_expanded = new_feature_tensor.expand(N, L, 1)

#     # 11. 원본 텐서와 새 피처 텐서를 마지막 차원(dim=2) 기준으로 합칩니다.  [55718, 8, 8]
#     final_tensor = torch.cat((tensor_data, new_feature_tensor_expanded), dim=2)

#     # 기본 인덱스 리스트 생성
#     # [0, 1, 2, 3, 4, 5, 6, 7]
#     indices = list(range(C_in+1))

#     # 마지막 두 인덱스의 자리를 바꿈
#     indices[C_in-1], indices[C_in] = indices[C_in], indices[C_in-1]

#     # 새로운 인덱스 순서 확인
#     # [0, 1, 2, 3, 4, 6, 5]

#     # 텐서의 마지막 차원(...)에 인덱스 리스트를 적용
#     # PyTorch가 이 순서대로 텐서를 재정렬하여 '복사본'을 만듭니다.
#     swapped_tensor = final_tensor[..., indices]

#     return swapped_tensor

# P2.5
def save_tensor_dataset(train_data, val_data, test_data, scaler, output_folder):
    print(f"\n데이터 shape 확인:")
    print(f"train_data: {train_data.shape} (파일 수 x 8 x 특성 수)")
    print(f"val_data: {val_data.shape} (파일 수 x 8 x 특성 수)")
    print(f"test_data: {test_data.shape} (파일 수 x 8 x 특성 수)")

    # scaler가 minmax인지 zscore인지 확인
    scaler_type = "minmax" if isinstance(scaler, MinMaxScaler) else "zscore"
    print(f"사용된 스케일러 타입: {scaler_type}")

    # 첫 번째 파일의 형태 확인
    print(f"\n첫 번째 학습 파일의 shape: {train_data[0].shape}")  # 256 x 6 이어야 함
    print("첫 번째 학습 파일의 처음 3개 행:")
    print(train_data[0][:3])  # 처음 3개 행의 값을 출력

    # 6. TensorDataset 생성 및 결과 저장
    os.makedirs(output_folder, exist_ok=True)

    # TensorDataset 생성 (향후 추가 레이블이나 메타데이터를 위해 확장 가능)
    train_dataset = TensorDataset(train_data)
    val_dataset = TensorDataset(val_data)
    test_dataset = TensorDataset(test_data)

    # 데이터셋 저장 (6:2:2 비율)
    torch.save(train_dataset, os.path.join(output_folder, "train_data.pt"))
    torch.save(val_dataset, os.path.join(output_folder, "val_data.pt"))
    torch.save(test_dataset, os.path.join(output_folder, "test_data.pt"))

    # 스케일러 저장
    with open(os.path.join(output_folder, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    # 처리된 데이터 통계 출력
    print("\n=== 처리 완료 ===")
    print(f"Train 데이터: {train_data.shape}")
    print(f"Val 데이터: {val_data.shape}")
    print(f"Test 데이터: {test_data.shape}")
    print(f"결과가 {output_folder}에 저장되었습니다.")
