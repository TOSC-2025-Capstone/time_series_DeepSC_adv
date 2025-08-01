import numpy as np
from scipy import interpolate
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import pdb
from .methods import *

"""
주요 변경사항:

rows_to_keep 마스크를 사용하여 이상치가 없는 행만 추적
모든 컬럼에 대해 이상치 검사 후, 하나라도 이상치가 있는 행은 제거
보간 관련 코드 제거

00001.csv (rows=480)
480 -> preprocess -> 256 (하나의 모델 단위 입력) -> 모델 처리 -> 256 -> 역보간 -> 480
비교 480 단위 (이상치가 제거된) 

상세한 통계 정보 출력:

각 컬럼별 이상치 수와 비율
전체 데이터에서 제거된 행의 수와 비율
정제 전후의 기본 통계 비교
"""


def detect_and_eliminate_outliers(df, threshold=3):
    """
    Z-score 기반으로 이상치(threshold 이상)를 탐지하고 해당 행을 제거
    """
    # 이상치가 있는 행을 추적하기 위한 마스크 초기화
    rows_to_keep = np.ones(len(df), dtype=bool)

    # 데이터 검증
    print("\n데이터프레임 정보:")
    print(f"데이터프레임 크기: {df.shape}")
    print("컬럼 목록:", df.columns.tolist())
    print("\n각 컬럼의 기본 통계:")
    print(df.describe())

    for column in df.columns:
        if column != "file_index":
            print(f"\n처리 중인 컬럼: {column}")

            # Z-score 계산
            mean_val = df[column].mean()
            std_val = df[column].std()
            print(f"평균: {mean_val:.4f}, 표준편차: {std_val:.4f}")

            # std가 0이면 모든 값이 같다는 의미이므로 처리가 필요없음
            if std_val == 0:
                continue

            z_scores = np.abs((df[column] - mean_val) / std_val)

            # 이상치 위치 파악
            outliers = z_scores > threshold

            print(f"Z-score 범위: {z_scores.min():.4f} ~ {z_scores.max():.4f}")
            print(f"임계값: {threshold}")

            if outliers.any():
                n_outliers = np.sum(outliers)
                outlier_percent = (n_outliers / len(df)) * 100
                print(
                    f"{column}에서 이상치 발견: {n_outliers}개 ({outlier_percent:.2f}%)"
                )

                # 이상치 값들의 통계
                outlier_values = df[column][outliers]
                print(
                    f"이상치 값 범위: {outlier_values.min():.4f} ~ {outlier_values.max():.4f}"
                )
                print(f"이상치 평균: {outlier_values.mean():.4f}")

                # 이상치가 있는 행 표시
                rows_to_keep[outliers] = False

                print(f"제거될 행의 수: {np.sum(~rows_to_keep)}개")

    # 이상치가 있는 행 한번에 제거
    df_cleaned = df[rows_to_keep].copy()

    print("\n이상치 제거 후 데이터프레임 정보:")
    print(f"원본 데이터 크기: {df.shape}")
    print(f"정제된 데이터 크기: {df_cleaned.shape}")
    print(f"제거된 행의 수: {len(df) - len(df_cleaned)}")
    print(f"제거된 비율: {((len(df) - len(df_cleaned)) / len(df)) * 100:.2f}%")

    print("\n정제된 데이터의 기본 통계:")
    print(df_cleaned.describe())

    return df_cleaned


def process_and_save_outlier_data(
    exclude_batteries=None, input_folder=None, output_folder=None, outlier_threshold=3
):
    """
    메인 함수: 이상치 탐지 및 제거 후 데이터프레임을 CSV로 저장
    """
    # 1. 메타데이터 로드 및 배터리 필터링
    discharge_data, discharge_files = load_and_filter_metadata(exclude_batteries)
    total_df = load_csv_data(discharge_files, input_folder)

    # 2. 이상치 탐지 및 제거
    print("이상치 탐지 및 제거 시작")
    df_cleaned = detect_and_eliminate_outliers(total_df, threshold=outlier_threshold)

    # 3. 데이터프레임 그룹화 후 csv로 저장
    print("데이터프레임 그룹화 및 저장 시작")
    count = df_cleaned["file_index"].nunique()
    print(f"총 {count}개의 파일 인덱스 발견")

    os.makedirs(output_folder, exist_ok=True)
    df_grouped = grouping_df(df_cleaned)
    for file_index, df in tqdm(df_grouped.items()):
        output_path = output_folder + f"{int(file_index):05d}.csv"
        df.to_csv(output_path, index=False)
        count -= 1

    if count == 0:
        print("outlier_eliminate.py, 모든 파일이 정상적으로 저장되었습니다.")

    # 3. 결과 시각화 자료 저장
    # file_index 컬럼을 제외한 feature_names 생성
    feature_names = [col for col in total_df.columns if col != "file_index"]
    visualize_data_comparison(
        total_df.drop(columns=["file_index"]),
        df_cleaned.drop(columns=["file_index"]),
        feature_names=feature_names,
        output_dir=f"cycle_preprocess/analysis/outlier_comparison_{outlier_threshold}/",
    )

    return df_cleaned, total_df


# if __name__ == "__main__":
# exclude_batteries = ["B0049", "B0050", "B0051", "B0052"]
# input_folder = "original_dataset/data/"
# output_folder = "cycle_preprocess/csv/outlier_cut/"
# df_cleaned, total_df = process_and_save_outlier_data(
#     exclude_batteries, input_folder, output_folder
# )
