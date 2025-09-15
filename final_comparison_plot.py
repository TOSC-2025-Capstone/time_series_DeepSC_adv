import os
import pdb

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from parameters.parameters import case_index


# 특정 케이스의 세 모델 복원 지표(MSE,MAE,RMSE)정보를 저장한 csv들을 받아서 plot하는 함수
def final_statistic_comparison_plot(csv_paths, case_labels=None, save_path=None):
    # 1. 데이터 읽기
    dfs = [pd.read_csv(path) for path in csv_paths]

    if case_labels is None:
        case_labels = [f"Case {i+1}" for i in range(len(dfs))]

    # 2. 지표 종류 정의
    metrics = ["MSE", "MAE", "RMSE"]

    # 3. Feature 순서 정의 (첫 번째 df 기준)
    features = dfs[0]["Feature"].unique()

    for metric in metrics:
        plt.figure(figsize=(12, 6))
        bar_width = 0.2
        x = np.arange(len(features))  # Feature별 위치

        for i, df in enumerate(dfs):
            # 각 feature에 대해 metric 값 추출
            values = []
            for feature in features:
                row = df[(df["Feature"] == feature) & (df["Metric"] == metric)]
                val = row["Mean"].values[0] if not row.empty else np.nan
                values.append(val)

            offset = (i - 1) * bar_width  # center 정렬
            plt.bar(x + offset, values, width=bar_width, label=case_labels[i])

        # 4. 그래프 꾸미기
        plt.xticks(x, features, rotation=45)
        plt.ylabel(f"{metric} (log scale)")
        plt.yscale("log")  # 값의 차이가 크므로 로그스케일 추천
        plt.ylim(1e-4, 1e6)
        plt.title(f"Feature-wise {metric} Comparison")
        plt.legend()
        plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
        plt.tight_layout()

        # 저장 또는 출력
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, f"{metric}_comparison.png"), dpi=300)
        else:
            plt.show()


# 특정 케이스의 세 모델에 대한 각 피쳐별 복원, 원본 사이클 비교
def plot_feature_comparison(
    original_path: str,
    deepsc_path: str,
    lstm_path: str,
    gru_path: str,
    feature_names: list,
    save_path: str,
    filename: str,
):
    # 1. 데이터 로드
    df_original = pd.read_csv(os.path.join(original_path, filename))
    df_deepsc = pd.read_csv(
        os.path.join(deepsc_path, filename.replace(".csv", "_reconstructed.csv"))
    )
    df_lstm = pd.read_csv(
        os.path.join(lstm_path, filename.replace(".csv", "_reconstructed.csv"))
    )
    df_gru = pd.read_csv(
        os.path.join(gru_path, filename.replace(".csv", "_reconstructed.csv"))
    )

    # 2. 시각화
    plt.figure(figsize=(18, 12))
    for i, feature in enumerate(feature_names):
        plt.subplot(2, 3, i + 1)
        plt.plot(df_original[feature], label="Original", linewidth=2)
        plt.plot(df_deepsc[feature], label="DeepSC", linestyle="--")
        plt.plot(df_lstm[feature], label="LSTM", linestyle="-.")
        plt.plot(df_gru[feature], label="GRU", linestyle=":")
        plt.title(feature)
        plt.xlabel("Timestep")
        plt.ylabel(feature)
        plt.legend()

    plt.tight_layout()

    os.makedirs(save_path, exist_ok=True)
    output_file = os.path.join(
        save_path, f"comparison_{filename.replace('.csv', '')}.png"
    )
    plt.savefig(output_file, dpi=300)
    plt.show()

    print(f"[✓] 저장 완료: {output_file}")


if __name__ == "__main__":
    # 예시 사용법
    prefix = "./results/performance_test"
    csv_paths = [
        prefix + "/case20.1.10/rayleigh_deepsc_MSE/performance_statistics.csv",
        prefix + "/case20.3/rayleigh_gru_MSE/performance_statistics.csv",
        prefix + "/case20.3.1/rayleigh_gru_MSE/performance_statistics.csv",
    ]
    # csv_paths = [
    #     prefix + "/case7.1/no_channel_deepsc_MSE/performance_statistics.csv",
    #     prefix + "/case8.1/no_channel_deepsc_MSE/performance_statistics.csv",
    #     prefix + "/case9.1/no_channel_deepsc_MSE/performance_statistics.csv",
    # ]

    filename = "01291.csv"
    save_path = (
        f"./final_comparison_plots/case{str(case_index).split('.')[0]}/{filename}/"
    )
    # case_labels = ["DeepSC", "GRU", "LSTM"]
    case_labels = ["20.1.10", "20.3", "20.3.1"]
    final_statistic_comparison_plot(
        csv_paths, case_labels, save_path=save_path
    )

    plot_feature_comparison(
        original_path="./cycle_preprocess/csv/outlier_cut/threshold_7/cycle_len_512",
        deepsc_path="./reconstruction/case20.1.10/reconstructed_rayleigh_deepsc_MSE",
        lstm_path="./reconstruction/case20.3/reconstructed_rayleigh_gru_MSE",
        gru_path="./reconstruction/case20.3.1/reconstructed_rayleigh_gru_MSE",
        # deepsc_path="./reconstruction/case7.1/reconstructed_no_channel_deepsc_MSE",
        # lstm_path="./reconstruction/case8.1/reconstructed_no_channel_deepsc_MSE",
        # gru_path="./reconstruction/case9.1/reconstructed_no_channel_deepsc_MSE",
        feature_names=[
            "Voltage_measured",
            "Current_measured",
            "Temperature_measured",
            "Current_load",
            "Voltage_load",
            "Time",
        ],
        save_path=save_path,
        filename=filename,
    )
