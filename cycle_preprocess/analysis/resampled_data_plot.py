import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pdb


def plot_all_features_distribution():
    # 데이터 폴더 경로
    data_dir = "./cycle_preprocess/csv/total_preprocessed"

    # 피처 이름 정의
    feature_names = [
        "Voltage_measured",
        "Current_measured",
        "Temperature_measured",
        "Current_load",
        "Voltage_load",
        "Time",
    ]

    # 결과 저장 폴더 생성
    save_dir = "./cycle_preprocess/analysis/feature_distribution_plots"
    os.makedirs(save_dir, exist_ok=True)

    # 모든 CSV 파일 목록 가져오기
    csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".csv")])

    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return

    print(f"Found {len(csv_files)} CSV files")

    # 각 피처별로 별도의 그래프 생성
    plt.figure(figsize=(20, 12))

    # 모든 파일의 데이터를 저장할 리스트 (피처별)
    all_feature_data = {feature: [] for feature in feature_names}

    # 모든 CSV 파일 읽기
    for file in tqdm(csv_files, desc="Reading CSV files"):
        file_path = os.path.join(data_dir, file)
        try:
            df = pd.read_csv(file_path)

            # 각 피처의 데이터 저장
            for feature in feature_names:
                if feature in df.columns:
                    all_feature_data[feature].append(df[feature].values)

        except Exception as e:
            print(f"Error reading {file}: {e}")

    # 각 피처별로 플롯 생성
    for idx, feature in enumerate(feature_names, 1):
        plt.subplot(2, 3, idx)

        # 무지개 색상 맵 생성 (파란색에서 보라색으로)
        num_cycles = len(all_feature_data[feature])  # 처음 100개 파일만
        colors = plt.cm.rainbow(np.linspace(0, 1, num_cycles))

        # 모든 파일의 해당 피처 데이터 플롯 (색상 그라데이션 적용)
        for idx, cycle_data in enumerate(all_feature_data[feature][:num_cycles]):
            plt.plot(cycle_data, alpha=0.4, color=colors[idx], linewidth=0.8)

        # 평균선 추가 (검정색으로 변경하여 더 잘 보이게)
        pdb.set_trace()
        mean_data = np.mean(all_feature_data[feature], axis=0)
        plt.plot(mean_data, color="black", linewidth=2, label="Mean", linestyle="--")

        plt.title(f"{feature} Distribution")
        plt.xlabel("Time Step (0-255)")
        plt.ylabel("Value")
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "all_features_distribution.png"), dpi=300)
    plt.close()

    # 각 피처별 상세 통계 저장
    stats_df = pd.DataFrame(columns=["Feature", "Mean", "Std", "Min", "Max"])

    for feature in feature_names:
        feature_data = np.array(all_feature_data[feature])
        stats = {
            "Feature": feature,
            "Mean": np.mean(feature_data),
            "Std": np.std(feature_data),
            "Min": np.min(feature_data),
            "Max": np.max(feature_data),
        }
        stats_df = pd.concat([stats_df, pd.DataFrame([stats])], ignore_index=True)

    # 통계 정보 저장
    stats_df.to_csv(os.path.join(save_dir, "feature_statistics.csv"), index=False)
    print(f"\n특성별 통계:")
    print(stats_df)

    # 추가 분석: 박스플롯으로 각 시점별 분포 시각화
    plt.figure(figsize=(20, 12))
    for idx, feature in enumerate(feature_names, 1):
        plt.subplot(2, 3, idx)

        # 시점별 박스플롯 데이터 준비 (10개 구간으로 나누어 표시)
        data = np.array(all_feature_data[feature])
        step_size = data.shape[1] // 10
        box_data = [data[:, i] for i in range(0, data.shape[1], step_size)]

        plt.boxplot(box_data)
        plt.title(f"{feature} Distribution by Time Steps")
        plt.xlabel("Time Step Groups")
        plt.ylabel("Value")
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "features_boxplot_distribution.png"), dpi=300)
    plt.close()

    print(f"\n결과가 {save_dir}에 저장되었습니다.")
    print("1. all_features_distribution.png - 모든 사이클의 피처별 시계열 그래프")
    print("2. features_boxplot_distribution.png - 피처별 시점 구간 분포")
    print("3. feature_statistics.csv - 피처별 통계 정보")


if __name__ == "__main__":
    plot_all_features_distribution()
