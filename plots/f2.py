import os
import pdb

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from parameters.parameters import case_index

# 한글 폰트 설정 - 영어로 대체
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False    # 음수 표시 깨짐 방지

# 전역 스타일 설정
plt.style.use("seaborn-v0_8-white")
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "figure.titlesize": 30,
        # "axes.titlesize": 24,
        "axes.labelsize": 24, # mse
        "xtick.labelsize": 20, # x 라벨
        "ytick.labelsize": 20,
        "legend.fontsize": 22,
        "axes.grid": False,
        "grid.alpha": 0.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        # 폰트/수식 설정 (마이너스 기호 경고 방지)
        "mathtext.fontset": "dejavusans",
        "mathtext.default": "regular",
        "text.usetex": False,
        "font.style": "normal",
    }
)


# 특정 케이스의 세 모델 복원 지표(MSE,MAE,RMSE)정보를 저장한 csv들을 받아서 plot하는 함수
def final_statistic_comparison_plot(csv_paths, case_labels=None, save_path=None):
    # 1. 데이터 읽기
    dfs = [pd.read_csv(path) for path in csv_paths]

    if case_labels is None:
        case_labels = [f"Case {i+1}" for i in range(len(dfs))]

    # 2. 지표 종류 정의 (MSE만 사용)
    metrics = ["MSE"]

    # 3. Feature 순서 정의 (첫 번째 df 기준) - Time 제외
    features = [f for f in dfs[0]["Feature"].unique() if f != "Time"]
    features_label = [f.replace("_", " ") for f in dfs[0]["Feature"].unique() if f != "Time"]

    features.append("Average")
    features_label.append("Average")

    # 색상 팔레트 (LSTM=밝은회색, GRU=중간회색, Transformer=진한회색, Inverted-Transformer=어두운회색, MI Net=검정)
    palette = ["#D9D9D9", "#BDBDBD", "#8C8C8C", "#696969", "#000000"]

    for metric in metrics:
        plt.figure(figsize=(32, 10))
        bar_width = 0.15
        # 피쳐별 간격을 위해 x 위치를 조정
        # x = np.arange(len(features)) * 0.85  # 피쳐간 간격 더 축소
        x = np.arange(len(features)) * 0.85  # 피쳐간 간격 더 축소

        for i, df in enumerate(dfs):
            avg_val = 0
            # 각 feature에 대해 metric 값 추출
            values = []
            for feature in features[:-1]:
                row = df[(df["Feature"] == feature) & (df["Metric"] == metric)]
                val = row["Mean"].values[0] if not row.empty else np.nan
                # accuracy 로 변환
                values.append(val)

            # LSTM 값 기준으로 비율 계산
            if i == 0:  # LSTM 기준
                lstm_values = values  # 나중에 나머지 모델 대비 비율 계산용
                normalized_values = [1.0 for _ in values]  # LSTM 자체는 1
            else:
                normalized_values = [v / lv if not np.isnan(v) else np.nan for v, lv in zip(values, lstm_values)]

            avg_relative = np.nanmean(normalized_values)
            normalized_values.append(avg_relative)

            # 같은 피쳐 내에서는 막대들이 붙어있도록 offset 조정
            offset = (i - (len(dfs) - 1) / 2) * bar_width
            color = palette[i % len(palette)]
            bars = plt.bar(
                x + offset,
                normalized_values,
                width=bar_width,
                label=case_labels[i],
                color=color,
                edgecolor="#333333",
                linewidth=0.6,
            )

            # 각 바 위에 비율 표시
            # for bar, val in zip(bars, normalized_values):
            #     plt.text(
            #         bar.get_x() + bar.get_width()/2,  # x 위치: 바 중앙
            #         bar.get_height() + 0.02,          # y 위치: 바 위쪽 약간 띄움
            #         f'{val:.2f}',                     # 소수점 2자리
            #         ha="center",
            #         va="bottom",
            #         fontsize=22,
            #         rotation=0,
            #     )
            # bars 그리기
            for j, val in enumerate(normalized_values):
                bar = plt.bar(
                    x[j] + offset,
                    val,
                    width=bar_width,
                    # label=case_labels[i] if j == 0 else "",  # 범례 중복 방지
                    color=color,
                    edgecolor="#333333",
                    linewidth=0.6,
                    hatch="////" if j == len(normalized_values) - 1 else None,
                )
                # 값 표시
                plt.text(
                    bar[0].get_x() + bar[0].get_width()/2,
                    bar[0].get_height() + 0.02,
                    f"{val:.2f}",
                    ha="center", va="bottom", fontsize=22
                )

        # 4. 그래프 꾸미기
        # x축 라벨을 막대들의 중앙에 맞춰서 배치
        plt.xticks(x, features_label, ha="center", rotation=0)
        plt.ylabel(f"{metric} (Compared to LSTM)")
        # plt.yscale("log")  # 값의 차이가 크므로 로그스케일 추천
        # y축 눈금 라벨 0도 고정 (크기는 rcParams 사용)
        plt.yticks(rotation=0)
        # 수학 표기(이탤릭) 대신 일반 문자열 포맷 사용
        ax = plt.gca()
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda y, pos: f"{y:g}")
        )
        # 최상단 제목(suptitle)과 그 아래 범례를 배치
        fig = plt.gcf()
        fig.suptitle(f"Feature-wise {metric} Comparison", y=0.98)
        fig.legend(
            # loc="upper center",
            # ncol=len(dfs),
            # bbox_to_anchor=(0.5, 0.955),
            frameon=True,
        )
        # 상단 영역(제목/범례) 확보
        plt.tight_layout(rect=[0, 0, 1, 0.90])

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
    # df_original = pd.read_csv(os.path.join(original_path, filename))
    # df_deepsc = pd.read_csv(
    #     os.path.join(deepsc_path, filename.replace(".csv", "_reconstructed.csv"))
    # )
    # df_lstm = pd.read_csv(
    #     os.path.join(lstm_path, filename.replace(".csv", "_reconstructed.csv"))
    # )
    # df_gru = pd.read_csv(
    #     os.path.join(gru_path, filename.replace(".csv", "_reconstructed.csv"))
    # )

    # 2. 시각화
    # plt.figure(figsize=(13, 10))
    # for i, feature in enumerate(feature_names):
    #     plt.subplot(2, 3, i + 1)
    #     plt.plot(df_original[feature], label="Original", linewidth=2)
    #     plt.plot(df_deepsc[feature], label="DeepSC", linestyle="--")
    #     plt.plot(df_lstm[feature], label="LSTM", linestyle="-.")
    #     plt.plot(df_gru[feature], label="GRU", linestyle=":")
    #     plt.title(feature, fontsize=22)
    #     plt.xlabel("Timestep")
    #     plt.ylabel(feature, fontsize=22)
    #     plt.legend()

    # plt.tight_layout()

    os.makedirs(save_path, exist_ok=True)
    output_file = os.path.join(
        save_path, f"comparison_{filename.replace('.csv', '')}.png"
    )
    plt.savefig(output_file, dpi=300)
    plt.show()

    print(f"[✓] 저장 완료: {output_file}")


def print_avg_mse_excluding_time(csv_paths, labels):
    # 각 CSV에서 Feature != 'Time'인 MSE의 Mean 평균을 계산하여 출력
    for path, label in zip(csv_paths, labels):
        metric_type = ""
        df = pd.read_csv(path)
        df = df[(df["Metric"] == "MSE") & (df["Feature"] != "Time")]
        avg = df["Mean"].mean()
        print(f"{label}: avg MSE (no Time) = {avg:.6f}")


if __name__ == "__main__":
    # 예시 사용법 (로컬에 존재하는 case20.* 경로로 업데이트)
    csv_paths = [
        # "results/performance_test/case39.2.3/Rayleigh_lstm_MSE/performance_statistics.csv",
        # "results/performance_test/case39.3.5/Rayleigh_gru_MSE/performance_statistics.csv",
        # "results/performance_test/case39.4.3/Rayleigh_deepsc_MSE/performance_statistics.csv",
        # "results/performance_test/case39.1.5/Rayleigh_deepsc_MSE/performance_statistics.csv",
        "results/performance_test/case44.2.1/Rayleigh_lstm_MSE/performance_statistics.csv",
        "results/performance_test/case44.3.1/Rayleigh_gru_MSE/performance_statistics.csv",
        "results/performance_test/case44.4.1/Rayleigh_deepsc_MSE/performance_statistics.csv",
        "results/performance_test/case44.1.4/Rayleigh_deepsc_MSE/performance_statistics.csv",
        "results/performance_test/case44.1.7/Rayleigh_deepsc_MSE/performance_statistics.csv",
    ]

    filename = "01291.csv"
    save_path = (
        f"./final_comparison_plots/case{str(case_index).split('.')[0]}_251015/압축15%+노이즈5db_4모델비교/{filename}/"
    )
    # case_labels = ["snr 5", "snr 10", "snr 15"]
    # case_labels = ["DeepSC", "GRU", "LSTM"]
    case_labels = ["LSTM", "GRU", "Transformer", "Inverted-Transformer-v1", "Inverted-Transformer-v2"]
    final_statistic_comparison_plot(
        csv_paths, case_labels, save_path=save_path
    )

    # 모델별 평균 MSE (Feature != 'Time') 출력
    print_avg_mse_excluding_time(csv_paths, case_labels)
