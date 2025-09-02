import os
import pdb

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import TensorDataset
from tqdm import tqdm

from .methods import *

"""
Hybrid Outlier Elimination:
1. Current 관련 피쳐 (Current_measured, Current_load 등): 카테고리컬 범위 기반 제거
2. 나머지 피쳐들: Z-score 기반 제거

- detect_current_outliers(): Current 관련 피쳐의 특정 범위(-4 ~ -2.5) 데이터 제거
- detect_and_eliminate_outliers(): 기존 Z-score 기반 이상치 제거 (Current 피쳐 제외)
- 두 단계를 순차적으로 적용하는 hybrid 접근법
"""


def detect_current_outliers(df, current_columns=None, remove_groups=None):
    """
    Current 관련 피쳐에서 특정 그룹의 데이터를 카테고리컬하게 제거
    첫 번째 코드의 로직을 따라 -3 group, -4 group을 제거

    Args:
        df: 입력 데이터프레임
        current_columns: Current 관련 컬럼명 리스트 (기본값: ['Current_measured', 'Current_load'])
        remove_groups: 제거할 그룹 정의 딕셔너리
                      기본값: {'-3_group': (-3.5, -2.5), '-4_group': (-4.0, -3.5)}

    Returns:
        df_filtered: Current 이상치가 제거된 데이터프레임
        removed_stats: 제거 통계 정보
    """

    if current_columns is None:
        current_columns = ["Current_measured", "Current_load"]

    if remove_groups is None:
        remove_groups = {
            "-3_group": (-3.5, -2.5),  # -3.5 <= Current_measured < -2.5
            "-4_group": (-4.0, -3.5),  # -4.0 < Current_measured < -3.5
        }

    print("=" * 70)
    print("CURRENT OUTLIER REMOVAL (Categorical Group-based)")
    print("=" * 70)

    # 현재 데이터에 존재하는 current 컬럼만 필터링
    existing_current_columns = [col for col in current_columns if col in df.columns]

    if not existing_current_columns:
        print("Warning: No current-related columns found in data")
        return df.copy(), {}

    print(f"Target current columns: {existing_current_columns}")
    print(f"Remove groups: {remove_groups}")

    original_rows = len(df)
    rows_to_keep = np.ones(len(df), dtype=bool)
    removal_stats = {}

    for col in existing_current_columns:
        col_removal_stats = {}
        col_total_removed = 0

        for group_name, (range_min, range_max) in remove_groups.items():
            group_mask = None

            if col == "Current_measured":
                if group_name == "-3_group":
                    # -3 group: -3.5 <= Current_measured < -2.5
                    group_mask = (df[col] >= range_min) & (df[col] < range_max)
                elif group_name == "-4_group":
                    # -4 group: -4.0 < Current_measured < -3.5
                    group_mask = (df[col] > range_min) & (df[col] < range_max)
                else:
                    # 사용자 정의 그룹의 경우
                    group_mask = (df[col] >= range_min) & (df[col] < range_max)

            if col == "Current_load":
                if group_name == "-3_group":
                    # -3 group: 3.5 > Current_load >= 2.5
                    group_mask = (df[col] < range_min * -1) & (
                        df[col] >= range_max * -1
                    )
                elif group_name == "-4_group":
                    # -4 group: 4.0 > Current_load >= 3.5
                    group_mask = (df[col] < range_min * -1) & (
                        df[col] >= range_max * -1
                    )
                else:
                    # 사용자 정의 그룹의 경우
                    group_mask = (df[col] >= range_min * -1) & (
                        df[col] < range_max * -1
                    )

            removed_count = group_mask.sum()
            col_total_removed += removed_count

            # 해당 범위의 데이터를 제거 대상으로 마킹
            rows_to_keep[group_mask] = False

            col_removal_stats[group_name] = {
                "removed_count": removed_count,
                "range": (range_min, range_max),
                "removal_percentage": (removed_count / original_rows) * 100,
            }

            if removed_count > 0:
                print(
                    f"  {col} {group_name}: Removed {removed_count} rows in range ({range_min}, {range_max})"
                )

        removal_stats[col] = {
            "groups": col_removal_stats,
            "total_removed": col_total_removed,
            "total_percentage": (col_total_removed / original_rows) * 100,
        }

    # 필터링된 데이터프레임 생성
    df_filtered = df[rows_to_keep].copy()

    total_removed = original_rows - len(df_filtered)

    print(f"\nCurrent Outlier Removal Summary:")
    print(f"Original rows: {original_rows:,}")
    print(f"Removed rows: {total_removed:,}")
    print(f"Remaining rows: {len(df_filtered):,}")
    print(f"Total removal percentage: {(total_removed/original_rows)*100:.2f}%")

    for col, stats in removal_stats.items():
        print(f"\n  {col} detailed stats:")
        for group_name, group_stats in stats["groups"].items():
            print(
                f"    {group_name}: {group_stats['removed_count']} rows ({group_stats['removal_percentage']:.2f}%)"
            )
        print(
            f"    Total for {col}: {stats['total_removed']} rows ({stats['total_percentage']:.2f}%)"
        )

    return df_filtered, removal_stats


def detect_and_eliminate_outliers_excluding_current(
    df,
    threshold=3,
    scale_scope="total",
    metadata_path=None,
    exclude_current_columns=True,
):
    """
    Z-score 기반 이상치 제거 (Current 컬럼 제외)

    Args:
        df: 입력 데이터프레임
        threshold: Z-score 임계값
        scale_scope: 스케일링 범위 ("total", "file", "battery_id")
        metadata_path: 메타데이터 경로 (scale_scope="battery_id"일 때 필요)
        exclude_current_columns: Current 관련 컬럼을 제외할지 여부
    """

    valid_scopes = ["total", "file", "battery_id"]
    if scale_scope not in valid_scopes:
        raise ValueError(f"scale_scope는 {valid_scopes} 중 하나여야 합니다.")

    print("=" * 70)
    print(f"Z-SCORE OUTLIER REMOVAL (threshold={threshold}, scope={scale_scope})")
    print("=" * 70)

    rows_to_keep = np.ones(len(df), dtype=bool)
    outlier_file_indices = set()
    reasons = []

    # Current 관련 컬럼 정의
    current_related_columns = [
        "Current_measured",
        "Current_load",
    ]

    # 처리할 컬럼 선택
    process_columns = []
    for col in df.columns:
        if col in ["file_index", "battery_id"]:
            continue
        if exclude_current_columns and col in current_related_columns:
            print(f"Excluding current-related column: {col}")
            continue
        process_columns.append(col)

    print(f"Processing columns: {process_columns}")

    for col in process_columns:
        # 스케일 범위별 평균/표준편차 계산
        if scale_scope == "file":
            mean_val = df.groupby("file_index")[col].transform("mean")
            std_val = df.groupby("file_index")[col].transform("std")
        elif scale_scope == "battery_id":
            # 배터리 단위 스케일링 준비
            if metadata_path is None:
                raise ValueError(
                    "scale_scope='battery_id'일 때는 metadata_path를 지정해야 합니다."
                )
            meta_df = pd.read_csv(metadata_path)

            # filename에서 숫자만 추출해 int로 변환 → file_index 생성
            if "filename" in meta_df.columns:
                meta_df["file_index"] = (
                    meta_df["filename"].str.extract(r"(\d+)").astype(int)
                )

            required_cols = {"file_index", "battery_id"}
            if not required_cols.issubset(meta_df.columns):
                raise ValueError(
                    f"metadata.csv에는 {required_cols} 컬럼이 반드시 포함되어야 합니다."
                )

            battery_map = meta_df.set_index("file_index")["battery_id"].to_dict()

            # df에 battery_id 컬럼 추가 (이미 있으면 스킵)
            if "battery_id" not in df.columns:
                df = df.copy()
                df["battery_id"] = df["file_index"].map(battery_map)

            mean_val = df.groupby("battery_id")[col].transform("mean")
            std_val = df.groupby("battery_id")[col].transform("std")
        else:  # "total"
            mean_val = df[col].mean()
            std_val = df[col].std()

        # 표준편차 0 방지
        if scale_scope in ["file", "battery_id"]:
            std_val = std_val.replace(0, np.nan)
            z = (df[col] - mean_val) / std_val
        else:  # total
            if std_val == 0 or np.isnan(std_val):
                print(f"Skipping column {col}: std=0 or NaN")
                continue
            z = (df[col] - mean_val) / std_val

        outliers = z.abs() > threshold
        if outliers.any():
            outlier_count = outliers.sum()
            rows_to_keep[outliers.values] = False
            outlier_file_indices.update(df.loc[outliers, "file_index"].unique())

            print(
                f"  {col}: Found {outlier_count} outliers ({(outlier_count/len(df))*100:.2f}%)"
            )

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

    # battery_id 컬럼 정리
    if scale_scope == "battery_id" and "battery_id" in df_cleaned.columns:
        df_cleaned = df_cleaned.drop(columns="battery_id")

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

    original_rows = len(df)
    removed_rows = original_rows - len(df_cleaned)

    print(f"\nZ-score Outlier Removal Summary:")
    print(f"Original rows: {original_rows:,}")
    print(f"Removed rows: {removed_rows:,}")
    print(f"Remaining rows: {len(df_cleaned):,}")
    print(f"Removal percentage: {(removed_rows/original_rows)*100:.2f}%")
    print(f"Outlier files: {len(outlier_file_indices)} files affected")

    return df_cleaned, sorted(outlier_file_indices), outlier_report


def process_and_save_hybrid_outlier_data(
    exclude_batteries=None,
    input_folder=None,
    output_folder=None,
    outlier_threshold=7,
    current_remove_groups=None,
    current_columns=None,
    scale_scope="battery_id",
    metadata_path="./original_dataset/metadata.csv",
):
    """
    하이브리드 이상치 제거 메인 함수:
    1. Current 관련 피쳐: 카테고리컬 그룹 기반 제거 (-3 group, -4 group)
    2. 나머지 피쳐들: Z-score 기반 제거

    Args:
        exclude_batteries: 제외할 배터리 ID 리스트
        input_folder: 입력 폴더 경로
        output_folder: 출력 폴더 경로
        outlier_threshold: Z-score 임계값
        current_remove_groups: Current 피쳐에서 제거할 그룹 정의
                            기본값: {'-3_group': (-3.5, -2.5), '-4_group': (-4.0, -3.5)}
        current_columns: Current 관련 컬럼명 리스트
        scale_scope: Z-score 스케일링 범위
        metadata_path: 메타데이터 파일 경로
    """

    print("HYBRID OUTLIER ELIMINATION PROCESS")
    print("=" * 70)
    print("Step 1: Current-based categorical removal")
    print("Step 2: Z-score based removal for remaining features")
    print("=" * 70)

    # 1. 메타데이터 로드 및 배터리 필터링
    discharge_data, discharge_files = load_and_filter_metadata(exclude_batteries)
    total_df = load_csv_data(discharge_files, input_folder)

    original_total_rows = len(total_df)
    print(f"Initial data loaded: {original_total_rows:,} rows")

    # 2. Current 관련 피쳐 이상치 제거 (1단계)
    df_after_current, current_removal_stats = detect_current_outliers(
        total_df, current_columns=current_columns, remove_groups=current_remove_groups
    )
    pdb.set_trace()

    # 3. 나머지 피쳐 Z-score 기반 이상치 제거 (2단계)
    df_cleaned, outlier_files, outlier_report = (
        detect_and_eliminate_outliers_excluding_current(
            df=df_after_current,
            threshold=outlier_threshold,
            scale_scope=scale_scope,
            metadata_path=metadata_path,
            exclude_current_columns=True,
        )
    )

    print("\n" + "=" * 70)
    print("HYBRID OUTLIER REMOVAL FINAL SUMMARY")
    print("=" * 70)
    print(f"Original rows: {original_total_rows:,}")
    print(f"After current removal: {len(df_after_current):,}")
    print(f"After z-score removal: {len(df_cleaned):,}")
    print(f"Total removed: {original_total_rows - len(df_cleaned):,}")
    print(
        f"Total removal rate: {((original_total_rows - len(df_cleaned))/original_total_rows)*100:.2f}%"
    )
    print(f"Affected files: {len(outlier_files)}")

    # outlier_report 저장
    if outlier_report is not None:
        os.makedirs("./analysis/outlier_reports", exist_ok=True)
        outlier_report.to_csv(
            f"./analysis/outlier_reports/hybrid_outlier_report_z={outlier_threshold}.csv",
            index=False,
        )
        print(f"Hybrid outlier report saved")

    # 4. 데이터프레임 그룹화 후 csv로 저장
    print("\n데이터프레임 그룹화 및 저장 시작")
    count = df_cleaned["file_index"].nunique()
    print(f"총 {count}개의 파일 인덱스 발견")

    os.makedirs(output_folder, exist_ok=True)
    df_grouped = grouping_df(df_cleaned)

    saved_count = 0
    for file_index, df in tqdm(df_grouped.items(), desc="Saving files"):
        output_path = os.path.join(output_folder, f"{int(file_index):05d}.csv")
        df.to_csv(output_path, index=False)
        saved_count += 1

    print(
        f"hybrid_outlier_elimination.py, {saved_count}개 파일이 정상적으로 저장되었습니다."
    )

    # 5. 결과 시각화 자료 저장
    feature_names = [col for col in total_df.columns if col != "file_index"]
    visualize_data_comparison(
        total_df.drop(columns=["file_index"]),
        df_cleaned.drop(columns=["file_index"]),
        feature_names=feature_names,
        output_dir=f"cycle_preprocess/analysis/hybrid_outlier_comparison_{outlier_threshold}/",
    )

    return (
        df_cleaned,
        total_df,
        {
            "current_removal_stats": current_removal_stats,
            "outlier_files": outlier_files,
            "outlier_report": outlier_report,
        },
    )


# 사용 예시
if __name__ == "__main__":
    exclude_batteries = ["B0049", "B0050", "B0051", "B0052"]
    input_folder = "original_dataset/data/"
    output_folder = "cycle_preprocess/csv/hybrid_outlier_cut/"

    # Current 피쳐에서 제거할 그룹 설정 (첫 번째 코드의 정확한 로직)
    current_remove_groups = {
        "-3_group": (-3.5, -2.5),  # -3.5 <= Current_measured < -2.5
        "-4_group": (-4.0, -3.5),  # -4.0 < Current_measured < -3.5
        # 더 넓은 -4 group을 원할 경우:
        # '-4_group': (-4.5, -3.5),  # -4.5 <= Current_measured < -3.5
    }

    # Current 관련 컬럼명
    current_columns = ["Current_measured", "Current_load"]

    df_cleaned, total_df, stats = process_and_save_hybrid_outlier_data(
        exclude_batteries=exclude_batteries,
        input_folder=input_folder,
        output_folder=output_folder,
        outlier_threshold=7,  # Z-score 임계값
        current_remove_groups=current_remove_groups,
        current_columns=current_columns,
        scale_scope="battery_id",
        metadata_path="./original_dataset/metadata.csv",
    )

    print("\n처리 완료!")
    print(f"Current 제거 통계:")
    for col, stats_col in stats["current_removal_stats"].items():
        print(
            f"  {col}: {stats_col['total_removed']} rows ({stats_col['total_percentage']:.2f}%)"
        )
    print(f"Z-score 이상치 파일 수: {len(stats['outlier_files'])}")
