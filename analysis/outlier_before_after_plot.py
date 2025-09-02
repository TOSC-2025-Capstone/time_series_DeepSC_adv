import os
import pdb
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from parameters.parameters import case_index, outlier_cut_csv_path

# 폴더 경로 (정제본/원본)
BASE_DIR = Path(__file__).parent
path_before = BASE_DIR / "../original_dataset/data"  # 원본
path_after = Path(outlier_cut_csv_path)  # 이상치 제거본
path_result = f"./analysis/outlier_comparison/{case_index}"
os.makedirs(path_result, exist_ok=True)
print(BASE_DIR, path_result)

feature_cols = [
    "Voltage_measured",
    "Current_measured",
    "Temperature_measured",
    "Current_load",
    "Voltage_load",
    "Time",
]

# 파일명 교집합
files_before = {f.name for f in path_before.glob("*.csv")}
files_after = {f.name for f in path_after.glob("*.csv")}
common_files = files_before & files_after

merged_data = {}


def row_key(df: pd.DataFrame, cols):
    """
    행 동일성 판정용 키.
    - 부동소수 오차 방지 위해 소수점 6자리 반올림 후 문자열 결합
    - 필요시 반올림 자리수 조절
    """
    rounded = df[cols].copy()
    for c in cols:
        rounded[c] = pd.to_numeric(rounded[c], errors="coerce").round(6)
    return rounded.astype(str).agg("|".join, axis=1)


removed_summary = []  # (fname, n_removed)


def main():
    for fname in sorted(common_files):
        # 주의: before=원본, after=정제본
        df_before = pd.read_csv(path_before / fname)[feature_cols].copy()  # 원본
        df_after = pd.read_csv(path_after / fname)[feature_cols].copy()  # 정제본

        # row_key를 사용한 정확한 비교
        df_before["row_key"] = row_key(df_before, feature_cols)
        df_after["row_key"] = row_key(df_after, feature_cols)

        # 삭제된 행 찾기 (원본 - 정제본)
        removed = df_before.merge(
            df_after[["row_key"]], on=["row_key"], how="left", indicator=True
        ).query("_merge == 'left_only'")[feature_cols]

        # --- Voltage_measured, Voltage_load 차이나는 경우만 필터링 ---
        removed = removed[
            removed["Voltage_measured"].notna() | removed["Voltage_load"].notna()
        ]

        n_removed = len(removed)
        if n_removed == 0:
            print(
                f"{fname}: No Voltage-related rows removed "
                f"(row counts - before: {len(df_before)}, after: {len(df_after)})"
            )
            continue

        removed_summary.append((fname, n_removed))
        print(f"{fname}: {n_removed} rows actually removed")

        # --- 플롯 (각 피쳐별로 전후 비교를 하나의 subplot에) ---
        plot_features = feature_cols[:-1]  # Time 제외한 전부
        num_features = len(plot_features)

        # 2 columns x ceil(num_features/2) rows 레이아웃으로 변경
        import math

        ncols = 2
        nrows = math.ceil(num_features / ncols)

        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(14, 4 * nrows), sharex=True
        )

        # axes를 1D 배열로 변환 (인덱싱 편의)
        if nrows == 1 and ncols == 1:
            axes = [axes]
        elif nrows == 1 or ncols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        for i, feature in enumerate(plot_features):
            ax = axes[i]

            # 공통 y한계: 원본 범위(이상치 포함)
            all_values = pd.concat(
                [df_before[feature], df_after[feature]], ignore_index=True
            )
            ymin, ymax = all_values.min(), all_values.max()
            margin = (ymax - ymin) * 0.05 if pd.notnull(ymax - ymin) else 0.0
            y0, y1 = ymin - margin, ymax + margin

            # 원본 데이터 (빨간색) - row 인덱스 사용
            ax.plot(
                range(len(df_before)),
                df_before[feature],
                label="Before (raw)",
                color="red",
                alpha=0.7,
                linewidth=1.2,
            )

            # 정제본 데이터 (파란색) - row 인덱스 사용
            ax.plot(
                range(len(df_after)),
                df_after[feature],
                label="After (clean)",
                color="blue",
                alpha=0.8,
                linewidth=1.2,
            )

            # 제거된 포인트들의 원본에서의 인덱스 찾기 (더 정확한 방법)
            if not removed.empty:
                # 원본 데이터프레임에 임시 인덱스 추가
                df_before_with_idx = df_before.copy()
                df_before_with_idx["original_idx"] = range(len(df_before_with_idx))

                # 제거된 행들을 원본과 매치하여 인덱스 찾기
                removed_with_key = removed.copy()
                removed_with_key["row_key"] = row_key(
                    removed_with_key, feature_cols[:-1]
                )

                df_before_with_key = df_before_with_idx.copy()
                df_before_with_key["row_key"] = row_key(
                    df_before_with_key, feature_cols[:-1]
                )

                # 매치되는 행들의 인덱스 찾기
                matched = removed_with_key.merge(
                    df_before_with_key[["row_key", "original_idx"]],
                    on="row_key",
                    how="left",
                )

                valid_matches = matched.dropna(subset=["original_idx"])
                if not valid_matches.empty:
                    removed_indices = valid_matches["original_idx"].astype(int).tolist()
                    ax.scatter(
                        removed_indices,
                        df_before.iloc[removed_indices][feature],
                        marker="x",
                        s=30,
                        color="black",
                        linewidths=2,
                        zorder=5,
                        label=f"Removed ({len(removed_indices)} pts)",
                        alpha=0.9,
                    )

            ax.set_ylim(y0, y1)
            ax.set_ylabel(feature)
            ax.set_title(f"{feature}")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=9)

        # 사용하지 않는 subplot 숨기기
        for i in range(num_features, len(axes)):
            axes[i].set_visible(False)

        # X라벨은 맨 아래 행에만
        for i in range(max(0, len(axes) - ncols), len(axes)):
            if axes[i].get_visible():
                axes[i].set_xlabel("Row Index")

        fig.suptitle(f"{fname}  |  Removed rows: {n_removed}", fontsize=14, y=0.995)
        plt.tight_layout()

        save_path = f"{path_result}/{fname.replace('.csv', '.png')}"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()  # 메모리 절약을 위해 figure 닫기

        print(f"  Saved: {save_path}")

    print(f"\n삭제된 행이 있는 파일 수: {len(removed_summary)}")
    for fname, n in removed_summary:
        print(f"{fname}: removed {n} rows")


if __name__ == "__main__":
    main()
