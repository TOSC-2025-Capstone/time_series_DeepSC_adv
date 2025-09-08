import os
import pdb
from glob import glob
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
from parameters.parameters import (
    case_index,
    eda_merged_path,
    eda_output_prefix_path,
    eda_readme_files,
    exclude_batteries,
    is_row_x_label_on_EDA,
    eda_target_type
)
import json

print(f"!!!!! {eda_target_type}")

# 컬러맵 설정
cmap = cm._colormaps["RdYlGn_r"]  # 초록 → 노랑 → 빨강

# 측정 컬럼 목록
measure_cols = [
    "Voltage_measured",
    "Current_measured",
    "Temperature_measured",
    "Current_load",
    "Voltage_load",
]

merged_path = eda_merged_path
output_prefix_path = eda_output_prefix_path
# outlier cut version
if eda_target_type == "outlier_cut":
    merged_path = f"./data/merged_outlier_cut/case_{case_index}/"
    output_prefix_path = f"./data/EDA_images_outlier_cut/case_{case_index}/"
# original version
elif eda_target_type == "original":
    merged_path = f"./data/merged/"
    output_prefix_path = f"./data/EDA_images/"

if is_row_x_label_on_EDA == True :
    output_prefix_path = os.path.join(output_prefix_path, "rows/")
elif is_row_x_label_on_EDA == False :
    output_prefix_path = os.path.join(output_prefix_path, "times/")

os.makedirs(merged_path, exist_ok=True)
os.makedirs(output_prefix_path, exist_ok=True)

# patterns = ["B0029", "B0030", "B0031", "B0032"]

# minmax 저장 파일 경로
scale_file = os.path.join("./data", "feature_minmax.json")

# 모든 배터리 csv 파일 경로
csv_files = {
    os.path.splitext(os.path.basename(f))[0]: f for f in glob(merged_path + "*.csv")
    # os.path.splitext(os.path.basename(f))[0]: f for f in glob(merged_path + f"{patterns}.csv")
}

# 딕셔너리 타입을 유지하면서 제외할 패턴 필터링
csv_files = {
    key: path for key, path in csv_files.items()
    if not any(pattern in key for pattern in exclude_batteries)
}

# 전체 데이터에서 각 피쳐의 최대/최소값을 미리 계산
print("전체 데이터의 스케일 범위 계산 중...")

# =============================
# 이상치 제거본 데이터일 경우 → 새로 계산 후 피쳐별 minmax 저장
# =============================
if eda_target_type == "outlier_cut":
    print("Original 데이터 → 피쳐별 min/max 계산 후 저장")
    all_data_for_scale = []
    for csv_path in csv_files.values():
        print(csv_path)
        df_temp = pd.read_csv(csv_path)
        all_data_for_scale.append(df_temp)

    if all_data_for_scale:
        df_scale = pd.concat(all_data_for_scale, ignore_index=True)

        feature_ranges = {}
        for col in measure_cols:
            if col in df_scale.columns:
                feature_ranges[col] = {
                    "min": float(df_scale[col].min()),
                    "max": float(df_scale[col].max()),
                }

        # JSON으로 저장
        with open(scale_file, "w") as f:
            json.dump(feature_ranges, f, indent=4)

        print(f"피쳐별 min/max 저장 완료 → {scale_file}")

    # Time 축의 최대/최소값도 계산
    time_range = {
        'min': df_scale["Time"].min(),
        'max': df_scale["Time"].max()
    }

# =============================
# 복원본 경우 → 파일 불러오기
# =============================
else:
    print("Original 데이터가 아님 → 저장된 min/max 불러오기")
    if os.path.exists(scale_file):
        with open(scale_file, "r") as f:
            feature_ranges = json.load(f)
        print(f"불러온 min/max 값: {scale_file}")
    else:
        raise FileNotFoundError(
            f"{scale_file} 파일이 없습니다. 먼저 eda_target_type = 'outlier_cut' 로 실행해 저장하세요."
        )

print("계산된 스케일 범위:")
for col in measure_cols:
    if col in feature_ranges:
        print(f"{col}: {feature_ranges[col]['min']:.2f} ~ {feature_ranges[col]['max']:.2f}")

for readme_path in eda_readme_files:
# for readme_path in ['./original_dataset/extra_infos\\README_29_30_31_32.txt']:
    # README 파일명에서 배터리 ID 추출
    basename = os.path.basename(readme_path)
    group_name = os.path.splitext(basename)[0]  # e.g., README_05_06_07_18
    id_part = group_name.replace("README_", "")
    battery_ids = id_part.split("_")
    print(f"README 처리 중: {basename} | 포함 배터리: {battery_ids}")

    # ["49", "50", "51", "52"] 형태로 변경
    transformed_list = [item[3:] for item in exclude_batteries]
    if set(battery_ids).issubset(set(transformed_list)):
        continue

    # 데이터프레임을 병합
    dfs = []
    for bid in battery_ids:
        csv_name = f"B{int(bid):04d}"  # 항상 B00NN 또는 B0NNN 포맷
        if csv_name in csv_files:
            df = pd.read_csv(csv_files[csv_name])
            df["battery_id"] = bid
            dfs.append(df)
        else:
            print(f"CSV 파일 없음: {csv_name}.csv")

    if not dfs:
        print(f"데이터 없음: {basename}")
        continue

    df_all = pd.concat(dfs, ignore_index=True)

    # LSTM, GRU 시퀀스 패딩 결과로 나오는 음수 time index 삭제
    df_all = df_all[df_all['Time'] >= 0 ]

    # 루프 안에서 group_name 기반으로 디렉토리 생성
    output_dir = f"{output_prefix_path}{group_name}"
    os.makedirs(output_dir, exist_ok=True)

    for col in measure_cols:
        if col not in feature_ranges:
            print(f"피쳐 {col}에 대한 스케일 정보가 없습니다. 건너뜁니다.")
            continue

        fig, ax = plt.subplots(figsize=(12, 5))
        norm = mcolors.Normalize(
            vmin=df_all["cycle_idx"].min(), vmax=df_all["cycle_idx"].max()
        )
        colors = plt.cm.tab10.colors

        for i, bid in enumerate(battery_ids):
            sub_df = df_all[df_all["battery_id"] == bid]
            for cycle_idx, group in sub_df.groupby("cycle_idx"):
                color_base = colors[i % len(colors)]
                alpha = 0.3 + 0.7 * norm(cycle_idx)
                if is_row_x_label_on_EDA == True :
                    ax.plot(
                    range(len(group)),
                    group[col],
                    color=color_base,
                    alpha=alpha,
                    # marker="o",
                    # markersize=3,
                    # linestyle="None",
                    label=(
                        f"B{bid}-C{cycle_idx}"
                        if cycle_idx == group["cycle_idx"].min()
                        else ""
                    ),
                )
                elif is_row_x_label_on_EDA ==False :
                    ax.plot(
                    group["Time"],
                    group[col],
                    color=color_base,
                    alpha=alpha,
                    # marker="o",
                    # markersize=3,
                    # linestyle="None",
                    label=(
                        f"B{bid}-C{cycle_idx}"
                        if cycle_idx == group["cycle_idx"].min()
                        else ""
                    ),
                )

        # x축과 y축 스케일을 전체 데이터의 최대/최소값으로 고정
        # ax.set_xlim(time_range['min'], time_range['max'])
        ax.set_ylim(feature_ranges[col]['min'], feature_ranges[col]['max'])

        ax.set_title(f"Battery Group: {id_part} {col} vs Time")
        ax.set_xlabel("Row Index" if is_row_x_label_on_EDA == True else "Time (s)")
        ax.set_ylabel(col)
        ax.grid(True)

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="best")

        plt.tight_layout(pad=2.0)

        save_path = os.path.join(output_dir, f"{group_name}_{col}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()

print("모든 README 기반 배터리 그룹 그래프 저장 완료")
