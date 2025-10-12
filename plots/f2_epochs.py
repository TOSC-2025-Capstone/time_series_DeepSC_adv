import os
import pdb

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from parameters.parameters import case_index

# 전역 스타일 설정
plt.style.use("seaborn-v0_8-white")
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "figure.titlesize": 20,
        # "axes.titlesize": 160,
        "axes.labelsize": 20,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "axes.grid": False,
        "grid.alpha": 0.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        # 폰트/수식 설정 (마이너스 기호 경고 방지)
        "font.family": "DejaVu Sans",
        "axes.unicode_minus": True,
        "mathtext.fontset": "dejavusans",
        "mathtext.default": "regular",
        "text.usetex": False,
        "font.style": "normal",
    }
)

def print_avg_mse_excluding_time(csv_paths, labels):
    # 각 CSV에서 Feature != 'Time'인 MSE의 Mean 평균을 계산하여 출력
    for path, label in zip(csv_paths, labels):
        df = pd.read_csv(path)
        df = df[(df["Metric"] == "MSE") & (df["Feature"] != "Time")]
        avg = df["Mean"].mean()
        print(f"{label}: avg MSE (no Time) = {avg:.6f}")


def print_avg_training_time(csv_paths, model_labels):
    """
    각 모델의 평균 학습시간을 계산하여 출력하는 함수

    Args:
        csv_paths: CSV 파일 경로 리스트
        model_labels: 모델 이름 리스트
    """
    print("\n=== 각 모델의 평균 학습시간 ===")
    for path, label in zip(csv_paths, model_labels):
        df = pd.read_csv(path)
        avg_time = df['Time(sec)'].mean()
        print(f"{label}: 평균 학습시간 = {avg_time:.2f}초")


def plot_epoch_time_comparison(csv_paths, model_labels, save_path=None):
    """
    각 모델의 epoch vs Time(sec) 그래프를 그리는 함수

    Args:
        csv_paths: CSV 파일 경로 리스트
        model_labels: 모델 이름 리스트
        save_path: 저장 경로 (선택사항)
    """
    # 1. 데이터 읽기
    dfs = [pd.read_csv(path) for path in csv_paths]

    # 2. 그래프 설정
    plt.figure(figsize=(16, 8))

    # 색상과 스타일 설
    colors = ["#D9D9D9", "#BDBDBD", "#8C8C8C", "#696969", "#000000"]
    linestyles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'D']

    # 3. 각 모델별로 그래프 그리기
    for i, (df, label) in enumerate(zip(dfs, model_labels)):
        plt.plot(
            df['Epoch'],
            df['Time(sec)'],
            color=colors[i % len(colors)],
            linestyle=linestyles[i % len(linestyles)],
            marker=markers[i % len(markers)],
            markersize=4,
            linewidth=2,
            label=label,
            alpha=0.8
        )

    # 4. 그래프 꾸미기 (크기는 rcParams 적용)
    plt.xlabel('Epoch')
    plt.ylabel('Time (sec)')

    # 최상단 제목(suptitle)과 그 아래 범례를 배치 (기존 MSE 그래프와 동일한 스타일)
    fig = plt.gcf()
    fig.suptitle("Training Time per Epoch Comparison", y=0.98)
    fig.legend(
        # loc="upper center",
        # ncol=len(model_labels),
        # bbox_to_anchor=(0.5, 0.955),
        frameon=True,
    )
    # 상단 영역(제목/범례) 확보
    plt.tight_layout(rect=[0, 0, 1, 0.90])

    # x축과 y축 범위 설정 (0부터 시작하지 않도록 수정)
    plt.xlim(left=0.5)  # 첫 번째 epoch(1)이 잘 보이도록 0.5부터 시작
    plt.ylim(bottom=0)

    # 5. 저장 또는 출력
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, "epoch_time_comparison.png"), dpi=300, bbox_inches='tight')
        print(f"[✓] 그래프 저장 완료: {os.path.join(save_path, 'epoch_time_comparison.png')}")
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    # Epoch vs Time 그래프를 위한 CSV 파일들
    csv_paths = [
        # "results/case21.2/rayleigh_lstm_MSE/epoch_stats.csv",
        # "results/case21.3/rayleigh_gru_MSE/epoch_stats.csv",
        # "results/case21.4/rayleigh_deepsc_MSE/epoch_stats.csv",
        # "results/case21.1/rayleigh_deepsc_MSE/epoch_stats.csv",
        # "results/case22.1/rayleigh_deepsc_MSE/epoch_stats.csv"
        "results/case33.2.1/AWGN_lstm_MSE/epoch_stats.csv",
        "results/case33.3.1/AWGN_gru_MSE/epoch_stats.csv",
        "results/case34.4.1/AWGN_deepsc_MSE/epoch_stats.csv",
        "results/case34.1.1/AWGN_deepsc_MSE/epoch_stats.csv",
    ]

    case_labels = ["LSTM", "GRU", "Transformer", "Inverted-Transformer", ]
    # case_labels = ["LSTM", "GRU", "Transformer", "Inverted-Transformer", "MI Net+Inverted-Transformer"]
    save_path = "./final_comparison_plots/epoch_time_comparison/"

    # 각 모델의 평균 학습시간 출력
    print_avg_training_time(csv_paths, case_labels)

    # Epoch vs Time 그래프 그리기
    print("\n=== Epoch vs Time 그래프 생성 중 ===")
    plot_epoch_time_comparison(csv_paths, case_labels, save_path=save_path)
