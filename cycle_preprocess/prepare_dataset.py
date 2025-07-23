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

"""
# prepare_dataset.py

detect_and_interpolate_outliers 함수:

Z-score 기반으로 이상치(threshold=3) 탐지
이상치를 선형 보간으로 대체
scale_data 함수:

MinMax 또는 Z-score 스케일링 수행
cycle_idx는 스케일링에서 제외
prepare_dataset 함수:

모든 resampled discharge 데이터를 하나의 DataFrame으로 통합
이상치 처리 및 보간 수행
선택한 방식으로 스케일링
파일 단위로 8:2 비율의 train/test 분할
결과를 pt 파일과 scaler.pkl로 저장

# 처리 순서
cycle_reshape.py 실행하여 resampled 데이터 생성
prepare_dataset.py 실행하여 전처리된 데이터셋 생성
preprocessed_data_check.py 실행하여 결과 비교

# 정규화 스케일러
Voltage, Current, Temperature 등 각 특성이 서로 다른 scale을 가지고 있더라도, 각각 독립적으로 정규화되어 처리됩니다.


# TensorDataset 관련:

torch.utils.data import TensorDataset 추가
단순 텐서 대신 TensorDataset으로 변환하여 저장
향후 레이블이나 메타데이터 추가를 위한 확장성 확보

# 이상치 제거 전후 비교 시각화 추가:

새로운 디렉토리 cycle_preprocess/analysis/outlier_comparison 생성

두 가지 비교 플롯 생성:
분포 비교 (히스토그램): outlier_removal_comparison.png
각 특성별로 원본 데이터와 이상치 제거 후 데이터의 분포 비교
시계열 비교: outlier_removal_timeseries.png
첫 번째 파일(256개 샘플)에 대해 원본과 이상치 제거 후 시계열 비교
각 특성별로 시간에 따른 변화 비교 가능
"""

# 한글깨짐 방지
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def detect_and_interpolate_outliers(df, threshold=3):
    """
    Z-score 기반으로 이상치(threshold 이상)를 탐지하고 선형 보간으로 대체
    """
    df_cleaned = df.copy()

    # pdb.set_trace()  # 디버깅용
    for column in df.columns:
        if column != "cycle_idx":
            # Z-score 계산
            mean_val = df[column].mean()
            std_val = df[column].std()

            # std가 0이면 모든 값이 같다는 의미이므로 처리가 필요없음
            if std_val == 0:
                continue

            z_scores = np.abs((df[column] - mean_val) / std_val)

            # 이상치 위치 파악
            outliers = z_scores > threshold

            if outliers.any():
                # 이상치 위치의 인덱스
                outlier_indices = np.where(outliers)[0]

                # 선형 보간을 위한 유효한 데이터 포인트
                valid_indices = np.where(~outliers)[0]

                # 유효한 값이 없는 경우 (모든 값이 이상치인 경우)
                if len(valid_indices) == 0:
                    print(f"경고: {column} - 모든 값이 이상치로 판단됨. 원래 값 유지.")
                    continue

                # 유효한 값이 하나뿐인 경우
                if len(valid_indices) == 1:
                    print(
                        f"경고: {column} - 유효한 값이 하나뿐입니다. 해당 값으로 채움."
                    )
                    df_cleaned.loc[outlier_indices, column] = df[column].iloc[
                        valid_indices[0]
                    ]
                    continue

                valid_values = df[column].iloc[valid_indices]

                try:
                    # 선형 보간 함수 생성
                    f = interpolate.interp1d(
                        valid_indices,
                        valid_values,
                        bounds_error=False,  # 범위를 벗어나는 외삽 허용
                        fill_value=(valid_values.iloc[0], valid_values.iloc[-1]),
                    )  # 끝점 처리

                    # 이상치를 보간된 값으로 대체
                    df_cleaned.loc[outlier_indices, column] = f(outlier_indices)
                except Exception as e:
                    print(
                        f"경고: {column} - 보간 중 오류 발생. 원래 값 유지. 오류: {str(e)}"
                    )
                    continue

    return df_cleaned


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


def prepare_dataset(scaler_type="minmax", test_ratio=0.2):
    """
    데이터셋 준비:
    1. 모든 discharge cycle 데이터를 하나의 DataFrame으로 통합
    2. 이상치 제거 및 보간
    3. 스케일링
    4. train/test 분할
    5. pt 파일로 저장
    """
    # 1. discharge 파일 목록 가져오기
    meta = pd.read_csv("original_dataset/metadata.csv")
    discharge_data = meta[meta["type"] == "discharge"]
    discharge_files = sorted(
        [f"{int(fname.split('.')[0]):05d}.csv" for fname in discharge_data["filename"]]
    )

    print(f"총 {len(discharge_files)}개의 discharge 파일 처리 시작")

    # 이상치 제거 전후 비교를 위한 디렉토리 생성
    outlier_plot_dir = "cycle_preprocess/analysis/outlier_comparison"
    os.makedirs(outlier_plot_dir, exist_ok=True)

    # 2. 모든 resampled discharge 데이터를 하나의 DataFrame으로 통합
    input_folder = "cycle_preprocess/analysis/resampled_256/"
    all_data = []
    missing_files = []

    print("\n데이터 파일 확인 중...")
    for fname in discharge_files:
        fpath = os.path.join(input_folder, fname)
        if not os.path.exists(fpath):
            missing_files.append(fname)

    if missing_files:
        print(f"\n경고: {len(missing_files)}개의 파일이 {input_folder}에 없습니다.")
        print("먼저 cycle_preprocess.py를 실행하여 resampled 데이터를 생성해주세요.")
        print("첫 번째 없는 파일:", missing_files[0])
        return None, None, None

    for fname in tqdm(discharge_files, desc="파일 로딩"):
        fpath = os.path.join(input_folder, fname)
        try:
            df = pd.read_csv(fpath)
            if len(df) > 0:  # 빈 데이터프레임이 아닌 경우만 추가
                all_data.append(df)
            else:
                print(f"\n경고: {fname}가 비어있습니다.")
        except Exception as e:
            print(f"\n오류: {fname} 로딩 중 문제 발생 - {str(e)}")

    if not all_data:
        print("\n오류: 로드된 데이터가 없습니다!")
        return None, None, None

    total_df = pd.concat(all_data, ignore_index=True)
    print(
        f"통합 데이터 크기: {total_df.shape} (파일 {len(all_data)}개 성공적으로 로드)"
    )

    # 3. 이상치 처리
    print("이상치 처리 중...")

    # 이상치 제거 전후 비교 플롯
    feature_names = [
        "Voltage_measured",
        "Current_measured",
        "Temperature_measured",
        "Current_load",
        "Voltage_load",
        "Time",
    ]

    print("이상치 제거 전후 비교 시각화 생성 중...")
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(total_df.columns):
        if col != "cycle_idx":
            plt.subplot(2, 3, i + 1)

            # 원본 데이터의 분포
            plt.hist(
                total_df[col], bins=50, alpha=0.5, label="원본 데이터", density=True
            )

            # 이상치 제거된 데이터
            cleaned_df = detect_and_interpolate_outliers(total_df)
            plt.hist(
                cleaned_df[col],
                bins=50,
                alpha=0.5,
                label="이상치 제거 후",
                density=True,
            )

            plt.title(f"{feature_names[i]} Distribution")
            plt.xlabel("Value")
            plt.ylabel("Density")
            plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(outlier_plot_dir, "outlier_removal_comparison.png"))
    plt.close()

    # 시계열 비교 (첫 번째 파일만)
    plt.figure(figsize=(15, 10))
    first_file_data = total_df.iloc[:256]  # 첫 번째 파일의 데이터
    first_file_cleaned = cleaned_df.iloc[:256]

    for i, col in enumerate(total_df.columns):
        if col != "cycle_idx":
            plt.subplot(2, 3, i + 1)
            plt.plot(first_file_data[col], alpha=0.7, label="원본 데이터")
            plt.plot(first_file_cleaned[col], alpha=0.7, label="이상치 제거 후")
            plt.title(f"{feature_names[i]} Time Series")
            plt.xlabel("Time Step")
            plt.ylabel("Value")
            plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(outlier_plot_dir, "outlier_removal_timeseries.png"))
    plt.close()

    print(f"이상치 제거 전후 비교 결과가 {outlier_plot_dir}에 저장되었습니다.")

    # 4. 스케일링
    print(f"{scaler_type} 스케일링 적용 중...")
    scaled_df, scaler = scale_data(cleaned_df, scaler_type)

    # 5. train/test 분할 (파일 단위로)
    n_files = len(discharge_files)
    n_samples_per_file = 256  # resampled length

    # 인덱스 생성
    indices = np.arange(len(scaled_df))
    test_size = int(n_files * test_ratio) * n_samples_per_file

    # 무작위로 test 파일 선택
    np.random.seed(42)
    test_indices = np.random.choice(
        np.arange(0, n_files) * n_samples_per_file,
        size=int(n_files * test_ratio),
        replace=False,
    )
    test_indices = np.concatenate(
        [np.arange(idx, idx + n_samples_per_file) for idx in test_indices]
    )

    # train/test 마스크 생성
    is_test = np.zeros(len(scaled_df), dtype=bool)
    is_test[test_indices] = True

    # 데이터 분할 및 reshape (각 파일을 256x6 형태로)
    n_features = len(scaled_df.columns)

    # train data를 256x6 형태의 텐서들로 변환
    train_samples = scaled_df[~is_test].values
    n_train_files = len(train_samples) // n_samples_per_file
    train_data = torch.FloatTensor(train_samples).view(
        n_train_files, n_samples_per_file, n_features
    )

    # test data를 256x6 형태의 텐서들로 변환
    test_samples = scaled_df[is_test].values
    n_test_files = len(test_samples) // n_samples_per_file
    test_data = torch.FloatTensor(test_samples).view(
        n_test_files, n_samples_per_file, n_features
    )

    print(f"\n데이터 shape 확인:")
    print(f"train_data: {train_data.shape} (파일 수 x 256 x 특성 수)")
    print(f"test_data: {test_data.shape} (파일 수 x 256 x 특성 수)")

    # 첫 번째 파일의 형태 확인
    print(f"\n첫 번째 학습 파일의 shape: {train_data[0].shape}")  # 256 x 6 이어야 함
    print("첫 번째 학습 파일의 처음 3개 행:")
    print(train_data[0][:3])  # 처음 3개 행의 값을 출력

    # 6. TensorDataset 생성 및 결과 저장
    output_folder = f"cycle_preprocess/preprocessed/processed_{scaler_type}/"
    os.makedirs(output_folder, exist_ok=True)

    # TensorDataset 생성 (향후 추가 레이블이나 메타데이터를 위해 확장 가능)
    train_dataset = TensorDataset(train_data)
    test_dataset = TensorDataset(test_data)

    # 데이터셋 저장
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

    return train_data, test_data, scaler


if __name__ == "__main__":
    print("처리 순서:")
    print("1. cycle_preprocess.py를 실행하여 resampled 데이터 생성")
    print("2. prepare_dataset.py를 실행하여 전처리된 데이터셋 생성")
    print("3. preprocessed_data_check.py를 실행하여 결과 비교")

    # MinMax 스케일링으로 데이터셋 준비
    print("\n=== MinMax Scaling ===")
    train_data_minmax, test_data_minmax, scaler_minmax = prepare_dataset(
        scaler_type="minmax"
    )

    if train_data_minmax is None:
        print("MinMax 스케일링 처리 실패. 프로그램을 종료합니다.")
        exit()

    # Z-score 스케일링으로 데이터셋 준비
    print("\n=== Z-score Scaling ===")
    train_data_zscore, test_data_zscore, scaler_zscore = prepare_dataset(
        scaler_type="zscore"
    )

    if train_data_zscore is None:
        print("Z-score 스케일링 처리 실패. 프로그램을 종료합니다.")
        exit()

    print("\n모든 처리가 완료되었습니다!")
    print("이제 preprocessed_data_check.py를 실행하여 결과를 비교해보세요.")
