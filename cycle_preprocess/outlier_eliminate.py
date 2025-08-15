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


def detect_and_eliminate_outliers(df, threshold=3, by_file=False):
    """
    by_file=True면 파일(file_index)별 평균/표준편차 기반 로컬 Z-score 계산
    False면 전체 데이터 전역 통계 기반 글로벌 Z-score 계산
    """

    rows_to_keep = np.ones(len(df), dtype=bool)
    outlier_file_indices = set()
    reasons = []

    for col in df.columns:
        if col == "file_index":
            continue

        if by_file:
            # 파일별 평균/표준편차
            mean_val = df.groupby("file_index")[col].transform("mean")
            std_val = df.groupby("file_index")[col].transform("std")
        else:
            mean_val = df[col].mean()
            std_val = df[col].std()

        # std==0 방지
        if by_file:
            std_val = std_val.replace(0, np.nan)
            z = (df[col] - mean_val) / std_val
        else:
            if std_val == 0 or np.isnan(std_val):
                continue
            z = (df[col] - mean_val) / std_val

        outliers = z.abs() > threshold

        if outliers.any():
            rows_to_keep[outliers.values] = False
            # ★ 해당 컬럼에서 이상치가 나온 file_index 추적
            outlier_files = df.loc[outliers, "file_index"].unique()
            outlier_file_indices.update(outlier_files)

            reasons.append(
                pd.DataFrame(
                    {
                        "row_idx": df.index[outliers],
                        "file_index": df.loc[outliers, "file_index"].values,
                        "column": col,
                        "value": df.loc[outliers, col].values,
                        "zscore": z.loc[outliers].values,
                    }
                )
            )

    df_cleaned = df[rows_to_keep].copy()

    outlier_report = None
    if reasons:
        reason_df = pd.concat(reasons, ignore_index=True)
        outlier_report = reason_df.groupby(
            ["row_idx", "file_index"], as_index=False
        ).agg(
            trigger_cols=("column", list),
            max_abs_z=("zscore", lambda s: float(np.nanmax(np.abs(s)))),
            any_z=("zscore", list),
        )

    print("\n이상치 제거 후 데이터프레임 정보:")
    print(f"\n이상치가 발생한 파일 index 목록: {sorted(outlier_file_indices)}")
    print(f"총 {len(outlier_file_indices)}개 파일에서 이상치 발견됨")
    print(f"원본 데이터 크기: {df.shape}")
    print(f"정제된 데이터 크기: {df_cleaned.shape}")
    print(f"제거된 행의 수: {len(df) - len(df_cleaned)}")
    print(f"제거된 비율: {((len(df) - len(df_cleaned)) / len(df)) * 100:.2f}%")

    return df_cleaned, sorted(outlier_file_indices), outlier_report


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
    # df_cleaned = detect_and_eliminate_outliers(total_df, threshold=outlier_threshold)
    df_cleaned, outlier_files, outlier_report = detect_and_eliminate_outliers(
        total_df, threshold=outlier_threshold
    )
    print("이상치 제거된 파일 index:", outlier_files)

    # outlier_report 저장
    if outlier_report is not None:
        os.makedirs("./analysis/outlier_reports", exist_ok=True)
        outlier_report.to_csv(
            f"./analysis/outlier_reports/outlier_report_by_z={outlier_threshold}.csv",
            index=False,
        )
        print(f"Outlier report saved: outlier_report.csv")

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
