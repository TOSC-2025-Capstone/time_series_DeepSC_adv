import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import json

from parameters.parameters import case_index

# 한글 폰트 설정 - 영어로 대체
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 전역 스타일 설정
plt.style.use("seaborn-v0_8-white")
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "figure.titlesize": 30,
        "axes.labelsize": 24,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 22,
        "axes.grid": False,
        "grid.alpha": 0.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "mathtext.fontset": "dejavusans",
        "mathtext.default": "regular",
        "text.usetex": False,
        "font.style": "normal",
    }
)


def load_feature_statistics(stats_path="data/feature_statistics.json", fallback_path="data/feature_minmax.json"):
    """
    피처 통계 정보 로드
    """
    if os.path.exists(stats_path):
        with open(stats_path, "r", encoding="utf-8") as f:
            return json.load(f), True  # True = full stats available
    elif os.path.exists(fallback_path):
        print(f"⚠️  {stats_path} 파일이 없습니다. {fallback_path} 사용 (일부 지표만 계산 가능)")
        with open(fallback_path, "r", encoding="utf-8") as f:
            return json.load(f), False  # False = only min/max available
    else:
        raise FileNotFoundError(f"통계 파일을 찾을 수 없습니다: {stats_path} 또는 {fallback_path}")


def calculate_accuracy(mse, rmse, feature_stats, method='nrmse_range'):
    """
    단일 피처에 대한 정확도 계산

    Parameters:
    - mse: Mean Squared Error
    - rmse: Root Mean Squared Error
    - feature_stats: 해당 피처의 통계 정보 dict
    - method: 'nrmse_range', 'nrmse_std', 'r2', 'percentage_based'

    Returns:
    - accuracy (float or np.nan)
    """
    if method == 'nrmse_range':
        # 방법 1: NRMSE (Range 기반)
        max_val = feature_stats.get("max", np.nan)
        min_val = feature_stats.get("min", np.nan)

        if max_val is None or min_val is None:
            return np.nan

        value_range = float(max_val) - float(min_val)

        if np.isnan(value_range) or value_range == 0:
            return np.nan

        acc = 1.0 - (rmse / value_range)
        return float(np.clip(acc, 0.0, 1.0))

    elif method == 'nrmse_std':
        # 방법 2: NRMSE (Std 기반)
        std = feature_stats.get("std", np.nan)

        if std is None or np.isnan(std) or float(std) == 0:
            return np.nan

        acc = 1.0 - (rmse / float(std))
        return float(np.clip(acc, 0.0, 1.0))

    elif method == 'r2':
        # 방법 3: R² Score
        variance = feature_stats.get("variance", np.nan)

        if variance is None or np.isnan(variance) or float(variance) == 0:
            return np.nan

        acc = 1.0 - (mse / float(variance))
        # R²는 음수 가능하므로 클리핑 안함
        return float(acc)

    elif method == 'percentage_based':
        # 방법 4: 평균 대비 RMSE 비율
        mean = feature_stats.get("mean", np.nan)

        if mean is None or np.isnan(mean) or float(mean) == 0:
            return np.nan

        error_percentage = (rmse / abs(float(mean))) * 100
        acc = max(0, 100 - error_percentage) / 100
        return float(np.clip(acc, 0.0, 1.0))

    else:
        raise ValueError(f"Unknown accuracy method: {method}")


def final_statistic_comparison_plot(
    csv_paths,
    case_labels=None,
    save_path=None,
    accuracy_method='nrmse_range',
    show_std_error=False
):
    """
    특정 케이스의 세 모델 복원 지표 비교 플롯

    Parameters:
    - csv_paths: performance_statistics.csv 파일 경로 리스트
    - case_labels: 케이스별 라벨
    - save_path: 저장 경로
    - accuracy_method: 정확도 계산 방법
        - 'nrmse_range': 1 - RMSE/Range (기본값, feature_minmax.json만 있어도 작동)
        - 'nrmse_std': 1 - RMSE/Std (feature_statistics.json 필요)
        - 'r2': R² Score = 1 - MSE/Variance (feature_statistics.json 필요)
        - 'percentage_based': 1 - Error%/100 (feature_statistics.json 필요)
    - show_std_error: 표준편차 에러바 표시 여부
    """
    # 1. 데이터 읽기
    dfs = [pd.read_csv(path) for path in csv_paths]

    if case_labels is None:
        case_labels = [f"Case {i+1}" for i in range(len(dfs))]

    # 2. Feature 순서 정의 - Time 제외 + Temperature 제외
    features = [f for f in dfs[0]["Feature"].unique() if f != "Time"]
    features = [f for f in features if "temperature" not in f.lower()]

    # 3. Feature 통계 로드
    feature_stats, has_full_stats = load_feature_statistics()

    # 선택한 방법이 사용 가능한지 확인
    if not has_full_stats and accuracy_method != 'nrmse_range':
        print(f"⚠️  {accuracy_method} 방법은 feature_statistics.json이 필요합니다.")
        print(f"   feature_minmax.json만 있으므로 'nrmse_range' 방법으로 전환합니다.")
        accuracy_method = 'nrmse_range'

    # 4. 색상 팔레트
    palette = ["#D9D9D9", "#BDBDBD", "#8C8C8C", "#696969", "#000000"]

    # 5. 평균 Accuracy 계산
    plt.figure(figsize=(16, 8))
    x = np.arange(len(dfs))

    avg_accs = []
    std_accs = []

    for i, df in enumerate(dfs):
        accuracies = []

        for feature in features:
            # MSE와 RMSE 추출
            mse_row = df[(df["Feature"] == feature) & (df["Metric"] == "MSE")]
            rmse_row = df[(df["Feature"] == feature) & (df["Metric"] == "RMSE")]

            if mse_row.empty or rmse_row.empty:
                accuracies.append(np.nan)
                continue

            mse = float(mse_row["Mean"].values[0])
            rmse = float(rmse_row["Mean"].values[0])

            # 해당 피처의 통계 정보
            stats = feature_stats.get(feature, {})

            # 정확도 계산
            acc = calculate_accuracy(mse, rmse, stats, accuracy_method)
            accuracies.append(acc)

        # 평균 및 표준편차 계산
        valid_accs = [a for a in accuracies if not np.isnan(a)]
        avg_acc = float(np.mean(valid_accs)) if valid_accs else np.nan
        std_acc = float(np.std(valid_accs)) if valid_accs else 0.0

        avg_accs.append(avg_acc)
        std_accs.append(std_acc)

    # 6. 막대 그래프
    bars = plt.bar(
        x,
        avg_accs,
        width=0.6,
        color=[palette[i % len(palette)] for i in range(len(avg_accs))],
        edgecolor="#333333",
        linewidth=0.8,
        yerr=std_accs if show_std_error else None,
        capsize=5 if show_std_error else 0
    )

    # 7. 값 표기
    for bar, val, std in zip(bars, avg_accs, std_accs):
        if not np.isnan(val):
            y_pos = (val + std + 0.02) if show_std_error else (val + 0.02)
            label = f"{val:.3f}±{std:.3f}" if show_std_error else f"{val:.3f}"

            plt.text(
                bar.get_x() + bar.get_width()/2,
                y_pos,
                label,
                ha="center", va="bottom", fontsize=20
            )

    # 8. 축/레이블 설정
    plt.xticks(x, case_labels, rotation=0)

    # 방법별 ylabel
    ylabel_map = {
        'nrmse_range': 'Average Accuracy (1 - RMSE/Range)',
        'nrmse_std': 'Average Accuracy (1 - RMSE/Std)',
        'r2': 'R² Score (1 - MSE/Variance)',
        'percentage_based': 'Average Accuracy (1 - Error%/100)'
    }
    plt.ylabel(ylabel_map.get(accuracy_method, 'Accuracy'))

    # ylim 설정 (R²는 음수 가능)
    if accuracy_method == 'r2':
        plt.ylim(-0.1, 1.05)
    else:
        plt.ylim(0.0, 1.05)

    ax = plt.gca()
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, pos: f"{y:g}"))

    # 타이틀
    title_map = {
        'nrmse_range': 'NRMSE (Range)',
        'nrmse_std': 'NRMSE (Std)',
        'r2': 'R² Score',
        'percentage_based': 'Percentage-based'
    }
    plt.title(f"Average Accuracy by Case ({title_map.get(accuracy_method, accuracy_method)})")
    plt.tight_layout()

    # 9. 저장 또는 출력
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        filename = f"Accuracy_comparison_{accuracy_method}.png"
        plt.savefig(os.path.join(save_path, filename), dpi=300)
        print(f"[✓] 저장 완료: {os.path.join(save_path, filename)}")
    else:
        plt.show()

    plt.close()

    return avg_accs, std_accs


def print_avg_accuracy(csv_paths, labels, accuracy_method='nrmse_range'):
    """
    각 케이스별 평균 Accuracy 출력
    """
    # Feature 통계 로드
    feature_stats, has_full_stats = load_feature_statistics()

    if not has_full_stats and accuracy_method != 'nrmse_range':
        print(f"⚠️  {accuracy_method} 방법은 feature_statistics.json이 필요합니다.")
        print(f"   'nrmse_range' 방법으로 계산합니다.\n")
        accuracy_method = 'nrmse_range'

    print(f"\n{'='*60}")
    print(f"평균 정확도 ({accuracy_method})")
    print(f"{'='*60}")

    for path, label in zip(csv_paths, labels):
        df = pd.read_csv(path)
        # Time 제외 + Temperature 제외
        df = df[(df["Metric"] == "MSE") & (df["Feature"] != "Time")]
        df = df[df["Feature"].str.lower().str.contains("temperature") == False]

        acc_list = []
        for _, row in df.iterrows():
            feature = row["Feature"]
            mse = float(row["Mean"])

            # RMSE 계산
            rmse = np.sqrt(mse)

            stats = feature_stats.get(feature, {})
            acc = calculate_accuracy(mse, rmse, stats, accuracy_method)

            if not np.isnan(acc):
                acc_list.append(acc)

        avg_acc = float(np.mean(acc_list)) if acc_list else np.nan
        std_acc = float(np.std(acc_list)) if acc_list else 0.0

        if np.isnan(avg_acc):
            print(f"{label:30s}: nan")
        else:
            print(f"{label:30s}: {avg_acc:.6f} ± {std_acc:.6f}")


if __name__ == "__main__":
    csv_paths = [
        # "results/performance_test/case23.1.1/rayleigh_deepsc_MSE/performance_statistics.csv",
        # "results/performance_test/case25.2.1/rayleigh_deepsc_MSE/performance_statistics.csv",
        # "results/performance_test/case25.2.2/rayleigh_deepsc_MSE/performance_statistics.csv",
        # "results/performance_test/case25.2.3/rayleigh_deepsc_MSE/performance_statistics.csv",
        "results/performance_test/case24.1.0/rayleigh_deepsc_MSE/performance_statistics.csv",
        "results/performance_test/case24.1.1/rayleigh_deepsc_MSE/performance_statistics.csv",
        "results/performance_test/case24.1.2/rayleigh_deepsc_MSE/performance_statistics.csv",
        "results/performance_test/case24.1.3/rayleigh_deepsc_MSE/performance_statistics.csv",
        "results/performance_test/case24.5.1/rayleigh_deepsc_MSE/performance_statistics.csv",
        "results/performance_test/case24.5.2/rayleigh_deepsc_MSE/performance_statistics.csv",
        "results/performance_test/case24.5.3/rayleigh_deepsc_MSE/performance_statistics.csv",
        "results/performance_test/case24.3.1/rayleigh_gru_MSE/performance_statistics.csv",
        "results/performance_test/case24.3.2/rayleigh_gru_MSE/performance_statistics.csv",
        "results/performance_test/case24.3.3/rayleigh_gru_MSE/performance_statistics.csv",
    ]

    filename = "01291.csv"
    save_path = (
        f"./final_comparison_plots/case{str(case_index).split('.')[0]}_251002/{filename}/"
    )
    case_labels = ["case 24.1.0", "case 24.1.1", "case 24.1.2", "case 24.1.3", "case 24.5.1", "case 24.5.2", "case 24.5.3", "case 24.3.1","case 24.3.2","case 24.3.3"]
    # case_labels = ["no compress", "4/5", "3/5", "2/5", "1/5"]
    # case_labels = ["no compress", "3/4", "2/4", "1/4"]

    # 정확도 계산 방법 선택
    # 'nrmse_range': feature_minmax.json만 있어도 작동 (기본값)
    # 'nrmse_std': feature_statistics.json 필요
    # 'r2': feature_statistics.json 필요
    # 'percentage_based': feature_statistics.json 필요

    accuracy_method = 'nrmse_range'  # 여기서 변경
    show_std_error = True  # 표준편차 에러바 표시 여부

    # 플롯 생성
    avg_accs, std_accs = final_statistic_comparison_plot(
        csv_paths,
        case_labels,
        save_path=save_path,
        accuracy_method=accuracy_method,
        show_std_error=show_std_error
    )

    # 케이스별 평균 Accuracy 출력
    # print_avg_accuracy(csv_paths, case_labels, accuracy_method=accuracy_method)

    # 여러 방법으로 비교하고 싶다면:
    for method in ['nrmse_range', 'nrmse_std', 'r2', 'percentage_based']:
        print(f"\n{'='*70}")
        print(f"방법: {method}")
        print(f"{'='*70}")
        final_statistic_comparison_plot(
            csv_paths, case_labels, save_path,
            accuracy_method=method, show_std_error=True
        )
        print_avg_accuracy(csv_paths, case_labels, accuracy_method=method)
