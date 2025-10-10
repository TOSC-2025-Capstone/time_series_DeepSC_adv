import os
import pdb

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from parameters.parameters import case_index
from sklearn.metrics import mean_squared_error

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
    model_1_path: str,
    model_2_path: str,
    model_3_path: str,
    model_4_path: str,
    feature_names: list,
    save_path: str,
    filename: str,
):
    # 1. 데이터 로드
    df_original = pd.read_csv(os.path.join(original_path, filename))
    df_model_1 = pd.read_csv(
        os.path.join(model_1_path, filename.replace(".csv", "_reconstructed.csv"))
    )
    df_model_2 = pd.read_csv(
        os.path.join(model_2_path, filename.replace(".csv", "_reconstructed.csv"))
    )
    df_model_3 = pd.read_csv(
        os.path.join(model_3_path, filename.replace(".csv", "_reconstructed.csv"))
    )
    df_model_4 = pd.read_csv(
        os.path.join(model_4_path, filename.replace(".csv", "_reconstructed.csv"))
    )

    # 2. 시각화
    plt.figure(figsize=(18, 12))
    for i, feature in enumerate(feature_names):
        plt.subplot(2, 3, i + 1)
        plt.plot(df_original[feature], label="Original", linewidth=2)
        plt.plot(df_model_1[feature], label="InvertedTransformer", linestyle="--")
        plt.plot(df_model_2[feature], label="LSTM", linestyle="-.")
        plt.plot(df_model_3[feature], label="GRU", linestyle=":")
        plt.plot(df_model_4[feature], label="Transformer", linestyle=":")
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


def plot_feature_comparison_with_residual(
    original_path: str,
    model_1_path: str,
    model_2_path: str,
    model_3_path: str,
    model_4_path: str,
    feature_names: list,
    save_path: str,
    filename: str,
):
    # 1. 데이터 로드
    df_original = pd.read_csv(os.path.join(original_path, filename))
    df_model_1 = pd.read_csv(os.path.join(model_1_path, filename.replace(".csv", "_reconstructed.csv")))
    df_model_2 = pd.read_csv(os.path.join(model_2_path, filename.replace(".csv", "_reconstructed.csv")))
    df_model_3 = pd.read_csv(os.path.join(model_3_path, filename.replace(".csv", "_reconstructed.csv")))
    df_model_4 = pd.read_csv(os.path.join(model_4_path, filename.replace(".csv", "_reconstructed.csv")))

    models = {
        "InvertedTransformer": (df_model_1, "orange", "--"),
        "LSTM": (df_model_2, "green", "-."),
        "GRU": (df_model_3, "red", ":"),
        "Transformer": (df_model_4, "purple", (0, (3, 1, 1, 1))),
    }

    # -----------------------
    # 2. 원본 vs 복원 비교
    # -----------------------
    plt.figure(figsize=(18, 12))
    for i, feature in enumerate(feature_names):
        ax = plt.subplot(2, 3, i + 1)

        # 원본
        ax.plot(df_original[feature], label="Original", color="black", linewidth=2)

        # 모델별 복원
        for model_name, (df_model, color, style) in models.items():
            mse = mean_squared_error(df_original[feature], df_model[feature])
            ax.plot(
                df_model[feature],
                label=f"{model_name} (MSE={mse:.3f})",
                color=color,
                linestyle=style,
                linewidth=1.5,
                alpha=0.9
            )

        ax.set_title(feature, fontsize=12)
        ax.set_xlabel("Timestep")
        ax.set_ylabel(feature)
        ax.legend(fontsize=8)

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    output_file_comp = os.path.join(save_path, f"comparison_{filename.replace('.csv', '')}.png")
    plt.savefig(output_file_comp, dpi=300)
    plt.show()
    print(f"[✓] 저장 완료 (원본-복원): {output_file_comp}")

    # -----------------------
    # 3. Residual² 플롯
    # -----------------------
    plt.figure(figsize=(18, 12))
    for i, feature in enumerate(feature_names):
        ax = plt.subplot(2, 3, i + 1)

        for model_name, (df_model, color, style) in models.items():
            residual_sq = (df_original[feature] - df_model[feature]) ** 2
            ax.plot(residual_sq, label=model_name, color=color, linestyle=style, linewidth=1)

        ax.set_title(f"{feature} Residual²", fontsize=12)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Squared Error")
        ax.legend(fontsize=8)

    plt.tight_layout()
    output_file_resid = os.path.join(save_path, f"residual_{filename.replace('.csv', '')}.png")
    plt.savefig(output_file_resid, dpi=300)
    plt.show()
    print(f"[✓] 저장 완료 (Residual²): {output_file_resid}")

if __name__ == "__main__":
    # 예시 사용법
    prefix = "./results/performance_test"
    csv_paths = [
        prefix + "/case29.2.2/AWGN_lstm_MSE/performance_statistics.csv",
        prefix + "/case29.3.2/AWGN_gru_MSE/performance_statistics.csv",
        prefix + "/case29.4.2/AWGN_deepsc_MSE/performance_statistics.csv",
        prefix + "/case29.1.2/AWGN_deepsc_MSE/performance_statistics.csv",
    ]

    filename = "01291.csv"
    save_path = (
        f"./final_comparison_plots/case{str(case_index).split('.')[0]}/{filename}/"
    )
    # case_labels = ["DeepSC", "GRU", "LSTM"]
    case_labels = ["21.1", "21.2", "21.3", "21.4"]
    final_statistic_comparison_plot(
        csv_paths, case_labels, save_path=save_path
    )

    # plot_feature_comparison(
    # plot_feature_comparison_with_residual(
    #     original_path="./cycle_preprocess/csv/outlier_cut/threshold_7/cycle_len_512",
    #     model_1_path="./reconstruction/case21.1/reconstructed_rayleigh_deepsc_MSE",
    #     model_2_path="./reconstruction/case21.2/reconstructed_rayleigh_lstm_MSE",
    #     model_3_path="./reconstruction/case21.3/reconstructed_rayleigh_gru_MSE",
    #     model_4_path="./reconstruction/case21.4/reconstructed_rayleigh_deepsc_MSE",
    #     feature_names=[
    #         "Voltage_measured",
    #         "Current_measured",
    #         "Temperature_measured",
    #         "Current_load",
    #         "Voltage_load",
    #         # "Time",
    #     ],
    #     save_path=save_path,
    #     filename=filename,
    # )
