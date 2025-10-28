import os
import pdb
import textwrap # 자동 줄바꿈 사용

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from parameters.parameters import case_index

# 한글 폰트 설정 - 영어로 대체
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# === 싱글 컬럼 레이아웃용 스타일 조정 ===
plt.style.use("seaborn-v0_8-whitegrid") # 그리드 추가 (논문 스타일에 맞게 조정 가능)
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "figure.titlesize": 16,  # 싱글 컬럼에 맞게 조정
        "axes.titlesize": 16,   # Subplot 제목 크기
        "axes.labelsize": 14,   # 축 레이블 크기
        "xtick.labelsize": 12,  # X축 틱 라벨 크기 (회전 고려)
        "ytick.labelsize": 12,  # Y축 틱 라벨 크기
        "legend.fontsize": 11,  # 범례 크기
        "grid.color": "grey",
        "grid.linestyle": ":",
        "grid.linewidth": 0.5,
        # "grid.alpha": 0.5,      # 약간의 그리드 표시
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "black",
        "legend.frameon": True,
        "legend.edgecolor": "lightgray",
        "mathtext.fontset": "dejavusans",
        "mathtext.default": "regular",
        "text.usetex": False,
        "font.style": "normal",
    }
)
# 전문적인 색상 팔레트 확장
# palette = [
#     "#0173B2", "#DE8F05", "#029E73", "#D55E00",
#     "#CC78BC", "#CA9161", "#FBAFE4", "#949494",
#     "#ECE133", "#56B4E9", "#000000"
# ]
palette = ["#D9D9D9", "#BDBDBD", "#8C8C8C", "#696969", "#000000"]

# === 싱글 컬럼용 막대 그래프 플로팅 함수 ===
def plot_bars_single_column(csv_paths, case_labels=None, save_path=None, metric_type=None, wrap_width=10):
    """지표(MSE 등)에 대해 첫 번째 케이스 대비 상대적 성능을 싱글 컬럼 레이아웃 막대 차트로 시각화"""
    # 1. 데이터 읽기
    dfs = [pd.read_csv(path) for path in csv_paths]
    if case_labels is None:
        case_labels = [f"Case {i+1}" for i in range(len(dfs))]
    base_label = case_labels[0] # 비교 기준 모델 라벨

    # 2. 지표 정의
    metrics = [metric_type]

    # 3. Feature 순서 및 라벨 정의 (Time 제외, 줄바꿈 적용)
    features = [f for f in dfs[0]["Feature"].unique() if f != "Time"]
    features_label_raw = [f.replace("_", " ") for f in features]
    # textwrap 적용
    features_label = [textwrap.fill(label, width=wrap_width) for label in features_label_raw]
    # features.append("Average")
    # features_label.append("Average") # Average는 그대로

    for metric in metrics:
        # --- Figure 크기 변경 ---
        fig, ax = plt.subplots(figsize=(8, 6)) # 싱글 컬럼용 (가로 7, 세로 6 인치)
        bar_width = 0.18 # 막대 너비
        num_cases = len(dfs)
        x = np.arange(len(features)) # X축 위치 (피처 중앙)

        base_values = None
        all_handles = [] # 범례 핸들 저장용

        for i, df in enumerate(dfs):
            # 값 계산 및 정규화 (이전과 동일)
            values = []
            # for feature in features[:-1]:
            for feature in features:
                row = df[(df["Feature"] == feature) & (df["Metric"] == metric)]
                val = row["Mean"].values[0] if not row.empty else np.nan
                values.append(val)

            if i == 0:
                base_values = values
                normalized_values = [1.0] * len(values)
            else:
                normalized_values = []
                for v, bv in zip(values, base_values):
                    if bv is None or np.isnan(bv) or bv == 0: normalized_values.append(np.nan)
                    elif v is None or np.isnan(v): normalized_values.append(np.nan)
                    else: normalized_values.append(v / bv)

            # avg_relative = np.nanmean(normalized_values)
            # normalized_values.append(avg_relative)

            # 막대 위치 오프셋
            offset = (i - (num_cases - 1) / 2) * bar_width
            color = palette[i % len(palette)]

            # 막대 그리기
            bars = ax.bar(
                x + offset,
                normalized_values,
                width=bar_width,
                label=case_labels[i],
                color=color,
                edgecolor="black",
                linewidth=0.5,
            )
            # bars[-1].set_hatch("///") # Average 해칭

            # --- 막대 위 값 표시 (폰트 크기 줄임) ---
            ax.bar_label(bars, fmt='%.1f', padding=2, fontsize=8, rotation=0)

            if bars: all_handles.append(bars[0]) # 범례용 핸들 저장

        # 4. 그래프 꾸미기
        # --- X축 라벨 회전 및 정렬 ---
        ax.set_xticks(x)
        # ax.set_xticklabels(features_label, rotation=45, ha="right")
        ax.set_xticklabels(features_label)
        ax.set_ylabel(f"Relative {metric}\n(Ratio to {base_label})")

        # # 기준선 추가
        # baseline_handle = ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, label=f"{base_label} Baseline (1.0)")
        # all_handles.append(baseline_handle) # 범례용 핸들 추가

        # ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

        # --- 제목 변경 (ax.set_title 사용) ---
        ax.set_title(f"Feature-wise {metric} Comparison (Relative)")

        # --- 범례 위치 변경 (그림 우측 상단 내부) ---
        all_labels = case_labels + [f"{base_label} Baseline (1.0)"]
        ax.legend(handles=all_handles, labels=all_labels, loc='upper right', bbox_to_anchor=(1.0, 1.0))

        # --- 레이아웃 조정 (표준 tight_layout) ---
        plt.tight_layout()

        # 저장 또는 출력
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            # 파일명에 '_single_col' 추가
            save_file = os.path.join(save_path, f"{metric}_relative_bar_single_col.png")
            plt.savefig(save_file, dpi=300, bbox_inches='tight')
            print(f"[✓] 저장 완료 (싱글 컬럼 막대): {save_file}")
        else:
            plt.show()
        plt.close(fig)

def plot_line_comparison(csv_paths, case_labels=None, save_path=None, metric_type=None):
    # 1. 데이터 읽기
    dfs = [pd.read_csv(path) for path in csv_paths]
    if case_labels is None:
        case_labels = [f"Case {i+1}" for i in range(len(dfs))]

    # 2. 지표 정의
    metrics = [metric_type]

    # 3. Feature 순서 정의
    features = [f for f in dfs[0]["Feature"].unique() if f != "Time"]
    features_label = [f.replace("_", " ") for f in features]

    # 원본 라벨 생성
    features_label_raw = [f.replace("_", " ") for f in features]

    # 텍스트 자동 줄바꿈 적용 (예: 10글자 기준)
    wrap_width = 12
    features_label = [textwrap.fill(label, width=wrap_width) for label in features_label_raw]

    features.append("Average")
    features_label.append("Average")

    # 4. 스타일 정의
    palette = ["#D9D9D9", "#BDBDBD", "#8C8C8C", "#696969", "#404040", "#202020", "#000000"]
    markers = ['o', 's', '^', 'D', 'v', '*', 'X']
    linestyles = ['-', '--', '-.', ':', (0, (1, 1)), (0, (5, 5)), (0, (3, 1, 1, 1))]
    x = np.arange(len(features))

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(18, 12))
        lstm_values = None
        all_normalized_values = []

        # 5. 데이터 전처리 (정규화 값 계산)
        for i, df in enumerate(dfs):
            values = []
            for feature in features[:-1]:
                row = df[(df["Feature"] == feature) & (df["Metric"] == metric)]
                val = row["Mean"].values[0] if not row.empty else np.nan
                values.append(val)
            if i == 0:
                lstm_values = values
                normalized_values = [1.0] * len(values)
            else:
                normalized_values = []
                for v, lv in zip(values, lstm_values):
                    if lv is None or np.isnan(lv) or lv == 0:
                        normalized_values.append(np.nan)
                    elif v is None or np.isnan(v):
                        normalized_values.append(np.nan)
                    else:
                        normalized_values.append(v / lv)
            avg_relative = np.nanmean(normalized_values)
            normalized_values.append(avg_relative)
            all_normalized_values.append(normalized_values)

        # 6. 꺾은선 그래프 그리기
        for i, normalized_values in enumerate(all_normalized_values):
            color = palette[i % len(palette)]
            ax.plot(
                x,
                normalized_values,
                label=case_labels[i],
                color=color,
                marker=markers[i % len(markers)],
                linestyle=linestyles[i % len(linestyles)],
                linewidth=2.5,
                markersize=10
            )

        # 7. 그래프 꾸미기
        ax.set_xticks(x)
        ax.set_xticklabels(features_label, ha="center", rotation=0)
        ax.set_ylabel(f"{metric} (Compared to {case_labels[0]})")
        ax.axhline(y=1.0, color='black', linestyle=':', linewidth=1.5, label=f"{case_labels[0]} Baseline")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, pos: f"{y:g}"))
        ax.axvspan(x[-1] - 0.5, x[-1] + 0.5, color='gray', alpha=0.1, zorder=0, label='Average Region')
        fig.suptitle(f"Feature-wise {metric} Comparison (Line)", y=0.98)

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles=handles,
            labels=labels,
            loc="upper center",
            ncol=min(len(dfs) + 2, 3),
            bbox_to_anchor=(0.5, 0.955),
            frameon=True,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.90])

        # 8. 저장
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, f"{metric}_line_comparison.png"), dpi=300)
        else:
            plt.show()
        plt.close(fig)


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
        # 항상 LSTM 먼저 배치
        # "results/performance_test/case39.2.3/Rayleigh_lstm_MSE/performance_statistics.csv",
        # "results/performance_test/case39.3.5/Rayleigh_gru_MSE/performance_statistics.csv",
        # "results/performance_test/case39.4.3/Rayleigh_deepsc_MSE/performance_statistics.csv",
        # "results/performance_test/case39.1.5/Rayleigh_deepsc_MSE/performance_statistics.csv",

        # "results/performance_test/case46.2.1/Rayleigh_lstm_MSE/performance_statistics.csv",
        # "results/performance_test/case46.3.1/Rayleigh_gru_MSE/performance_statistics.csv",
        # "results/performance_test/case46.4.1/Rayleigh_deepsc_MSE/performance_statistics.csv",
        # "results/performance_test/case46.1.1/Rayleigh_deepsc_MSE/performance_statistics.csv",

        "results/performance_test/case55.1.1/Rayleigh_deepsc_MSE/performance_statistics.csv",
        "results/performance_test/case55.2.1/Rayleigh_lstm_MSE/performance_statistics.csv",
        "results/performance_test/case55.1.2/Rayleigh_deepsc_MSE/performance_statistics.csv",
        "results/performance_test/case55.2.2/Rayleigh_lstm_MSE/performance_statistics.csv",
        # "results/performance_test/case55.1.1/Rayleigh_deepsc_MSE/performance_statistics.csv",
        # "results/performance_test/case55.2.1/Rayleigh_lstm_MSE/performance_statistics.csv",
    ]

    # filename = "01291.csv"
    filename = "02619.csv"
    metric_type = "MSE"
    save_path = (
        f"./final_comparison_plots/case{str(case_index).split('.')[0]}_251028/{metric_type}/데이터셋8-1-1/{filename}/"
    )
    # case_labels = ["snr 5", "snr 10", "snr 15"]
    # case_labels = [ "Inverted-Transformer_k7" , "Inverted-Transformer_k3", "Inverted-Transformer_k5"]
    # case_labels = ["LSTM", "GRU", "Transformer", "Inverted-Transformer"]
    case_labels = [ "Inverted-Transformer", "LSTM", "Inverted-Transformer-21", "LSTM-21"]
    # case_labels = [ "no-compress_lstm", "layer 1", "layer 2", "layer 4", "layer 1", "layer 2", "layer 4"]

    # 1. 기본 라벨 정의
    # base_labels = ["no-compress",  "feature 3/5",  "feature 1/5"]
    # base_labels = ["no-compress", "sequence 3/4", "sequence 2/4", "sequence 1/4"]

    # # 2. 각 모델별 접두사 추가
    # labels_lstm = [f"LSTM_{label}" for label in base_labels]
    # labels_it = [f"InvertedTransformer_{label}" for label in base_labels]

    # # 3. 두 리스트 결합
    # case_labels = labels_lstm + labels_it

    # final_statistic_comparison_plot(
    plot_bars_single_column(
        csv_paths, case_labels, save_path=save_path, metric_type=metric_type, wrap_width=12
    )

    # plot_line_comparison(
    #     csv_paths, case_labels, save_path=save_path, metric_type=metric_type
    # )

    # 모델별 평균 MSE (Feature != 'Time') 출력
    print_avg_mse_excluding_time(csv_paths, case_labels)
