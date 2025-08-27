import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import pdb

# 폴더 경로 (정제본/원본)
BASE_DIR = Path(__file__).parent
path_before = (
    BASE_DIR / "../cycle_preprocess/csv/outlier_cut/threshold_7/cycle_len_512"
)  # 이상치 제거본
path_after = BASE_DIR / "../original_dataset/data"  # 원본
path_result = "./outlier_comparison/voltages"
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
# pdb.set_trace()  # 필요시 주석 해제


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

for fname in sorted(common_files):
    # 주의: before=정제본, after=원본
    df_before = pd.read_csv(path_before / fname)[feature_cols].copy()
    df_after = pd.read_csv(path_after / fname)[feature_cols].copy()

    # row_key를 사용한 정확한 비교
    df_before["row_key"] = row_key(df_before, feature_cols)
    df_after["row_key"] = row_key(df_after, feature_cols)

    # 삭제된 행 찾기 (원본 - 정제본)
    removed = df_after.merge(
        df_before[["row_key"]], on=["row_key"], how="left", indicator=True
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
        # 삭제된 행 없는 파일은 스킵

    removed_summary.append((fname, n_removed))
    print(f"{fname}: {n_removed} rows actually removed")

    # --- 플롯 (좌: 원본+마커 / 우: 정제본) ---
    # plot_features = [c for c in feature_cols if c != "Time"]
    plot_features = ["Voltage_measured", "Voltage_load"]
    num_features = len(plot_features)

    fig, axes = plt.subplots(
        nrows=num_features,
        ncols=2,
        figsize=(14, 3.6 * num_features),
        sharex="row",
        sharey="row",
    )

    # axes가 2D 보장되도록
    if num_features == 1:
        axes = [axes]  # [(ax_left, ax_right)]

    for i, feature in enumerate(plot_features):
        ax_left, ax_right = axes[i]

        # 공통 y한계: 원본 범위(이상치 포함)
        ymin, ymax = df_after[feature].min(), df_after[feature].max()
        margin = (ymax - ymin) * 0.05 if pd.notnull(ymax - ymin) else 0.0
        y0, y1 = ymin - margin, ymax + margin

        # --- 좌측: 원본 + 제거된 행 마커 ---
        ax_left.plot(
            df_after["Time"], df_after[feature], label="Before (raw)", color="red"
        )
        if not removed.empty:
            ax_left.scatter(
                removed["Time"],
                removed[feature],
                marker="x",
                s=28,
                linewidths=1.2,
                zorder=3,
                label="Removed outlier(s)",
            )
        ax_left.set_ylim(y0, y1)
        ax_left.set_ylabel(feature)
        ax_left.grid(True, alpha=0.3)
        if i == 0:
            ax_left.legend(loc="best")
        ax_left.set_title(f"{feature} — Raw (+markers)")

        # --- 우측: 정제본 ---
        ax_right.plot(
            df_before["Time"], df_before[feature], label="After (clean)", color="orange"
        )
        ax_right.set_ylim(y0, y1)
        ax_right.grid(True, alpha=0.3)
        if i == 0:
            ax_right.legend(loc="best")
        ax_right.set_title(f"{feature} — Clean")

    # X라벨은 맨 아래 행에만
    axes[-1][0].set_xlabel("Time")
    axes[-1][1].set_xlabel("Time")

    fig.suptitle(f"{fname}  |  removed rows: {n_removed}", y=0.995)
    plt.tight_layout()
    save_path = f"{path_result}/{fname.replace('.csv', '.png')}"
    plt.savefig(save_path, dpi=150)  # show 전에 저장
    # plt.show()

print(f"\n삭제된 행이 있는 파일 수: {len(removed_summary)}")
for fname, n in removed_summary:
    print(f"{fname}: removed {n} rows")
