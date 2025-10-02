import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import json
from parameters.parameters import case_index

def load_feature_statistics(stats_path="data/feature_statistics.json"):
    """
    피처 통계 정보 로드 (없으면 기본값 사용)
    """
    if os.path.exists(stats_path):
        with open(stats_path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        print(f"⚠️  {stats_path} 파일이 없습니다. feature_minmax.json 사용")
        with open("data/feature_minmax.json", "r", encoding="utf-8") as f:
            minmax_data = json.load(f)

        # min, max만 있는 경우 - std, variance 추정 불가능하므로 경고
        print("⚠️  std, variance 정보가 없어 nrmse_std, r2 계산 불가")
        return minmax_data


def calculate_accuracy_metrics(df, feature_stats, method='nrmse_range'):
    """
    여러 방법으로 Accuracy 계산
    """
    features = [f for f in df["Feature"].unique()
                if f != "Time" and "temperature" not in f.lower()]

    accuracies = []

    for feature in features:
        # MSE, RMSE 추출
        mse_row = df[(df["Feature"] == feature) & (df["Metric"] == "MSE")]
        rmse_row = df[(df["Feature"] == feature) & (df["Metric"] == "RMSE")]

        if mse_row.empty or rmse_row.empty:
            accuracies.append(np.nan)
            continue

        mse = float(mse_row["Mean"].values[0])
        rmse = float(rmse_row["Mean"].values[0])

        stats = feature_stats.get(feature, {})

        if method == 'nrmse_range':
            # 방법 1: Range 기반 (현재 방식)
            max_val = stats.get("max", np.nan)
            min_val = stats.get("min", np.nan)

            if max_val is None or min_val is None:
                value_range = np.nan
            else:
                value_range = float(max_val) - float(min_val)

            if np.isnan(value_range) or value_range == 0:
                acc = np.nan
            else:
                acc = 1.0 - (rmse / value_range)

        elif method == 'nrmse_std':
            # 방법 2: 표준편차 기반
            std = stats.get("std", np.nan)

            if std is None or np.isnan(std) or float(std) == 0:
                acc = np.nan
            else:
                acc = 1.0 - (rmse / float(std))

        elif method == 'r2':
            # 방법 3: R² Score 기반
            variance = stats.get("variance", np.nan)

            if variance is None or np.isnan(variance) or float(variance) == 0:
                acc = np.nan
            else:
                acc = 1.0 - (mse / float(variance))

        elif method == 'percentage_based':
            # 방법 4: 평균 대비 RMSE 비율
            mean = stats.get("mean", np.nan)

            if mean is None or np.isnan(mean) or float(mean) == 0:
                acc = np.nan
            else:
                error_percentage = (rmse / abs(float(mean))) * 100
                acc = max(0, 100 - error_percentage) / 100

        else:
            acc = np.nan

        # 정확도를 0~1 범위로 클리핑 (R²는 음수 허용)
        if not np.isnan(acc) and method != 'r2':
            acc = float(np.clip(acc, 0.0, 1.0))

        accuracies.append(acc)

    return accuracies, features


def final_statistic_comparison_plot_v2(
    csv_paths,
    case_labels=None,
    save_path=None,
    accuracy_method='nrmse_range',
    stats_path="data/feature_statistics.json"
):
    """
    개선된 정확도 비교 플롯
    """
    dfs = [pd.read_csv(path) for path in csv_paths]

    if case_labels is None:
        case_labels = [f"Case {i+1}" for i in range(len(dfs))]

    # Feature 통계 로드
    feature_stats = load_feature_statistics(stats_path)

    # 색상 팔레트
    palette = ["#D9D9D9", "#BDBDBD", "#8C8C8C", "#696969", "#000000"]

    # 평균 Accuracy 계산
    plt.figure(figsize=(16, 8))
    x = np.arange(len(dfs))

    avg_accs = []
    std_accs = []

    for df in dfs:
        accuracies, features = calculate_accuracy_metrics(df, feature_stats, accuracy_method)

        valid_accs = [a for a in accuracies if not np.isnan(a)]

        if valid_accs:
            avg_acc = float(np.mean(valid_accs))
            std_acc = float(np.std(valid_accs))
        else:
            avg_acc = np.nan
            std_acc = 0.0

        avg_accs.append(avg_acc)
        std_accs.append(std_acc)

    # nan 체크
    if all(np.isnan(a) for a in avg_accs):
        print(f"⚠️  {accuracy_method} 방법으로 계산된 정확도가 모두 NaN입니다.")
        print(f"   feature_statistics.json 파일에 필요한 통계 정보가 있는지 확인하세요.")
        return avg_accs, std_accs

    # 막대 그래프 + 에러바
    bars = plt.bar(
        x,
        avg_accs,
        width=0.6,
        color=[palette[i % len(palette)] for i in range(len(avg_accs))],
        edgecolor="#333333",
        linewidth=0.8,
        yerr=std_accs,
        capsize=5
    )

    # 값 표기
    for bar, val, std in zip(bars, avg_accs, std_accs):
        if not np.isnan(val):
            y_pos = val + std + 0.02 if not np.isnan(std) else val + 0.02
            label = f"{val:.3f}±{std:.3f}" if not np.isnan(std) else f"{val:.3f}"
            plt.text(
                bar.get_x() + bar.get_width()/2,
                y_pos,
                label,
                ha="center", va="bottom", fontsize=18
            )

    # 축 설정
    plt.xticks(x, case_labels, rotation=0)

    # 방법별 ylabel 설정
    ylabel_map = {
        'nrmse_range': 'Accuracy (1 - RMSE/Range)',
        'nrmse_std': 'Accuracy (1 - RMSE/Std)',
        'r2': 'R² Score (1 - MSE/Variance)',
        'percentage_based': 'Accuracy (1 - Error%/100)'
    }
    plt.ylabel(ylabel_map.get(accuracy_method, 'Accuracy'), fontsize=24)

    # R²는 음수 가능하므로 ylim 조정
    if accuracy_method == 'r2':
        plt.ylim(-0.1, 1.05)
    else:
        plt.ylim(0.0, 1.05)

    plt.title(f"Average Accuracy Comparison ({accuracy_method.upper()})", fontsize=28)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()

    # 저장
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        filename = f"Accuracy_comparison_{accuracy_method}.png"
        plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches='tight')
        print(f"[✓] 저장 완료: {os.path.join(save_path, filename)}")
    else:
        plt.show()

    plt.close()

    return avg_accs, std_accs


def compare_all_methods(csv_paths, case_labels, save_path=None, stats_path="data/feature_statistics.json"):
    """
    모든 정확도 계산 방법을 비교하는 함수
    """
    methods = ['nrmse_range', 'nrmse_std', 'r2', 'percentage_based']

    results = {}
    for method in methods:
        print(f"\n{'='*50}")
        print(f"계산 중: {method}")
        print(f"{'='*50}")

        avg_accs, std_accs = final_statistic_comparison_plot_v2(
            csv_paths, case_labels, save_path, method, stats_path
        )
        results[method] = {'avg': avg_accs, 'std': std_accs}

    # 결과 비교 테이블 출력
    print("\n" + "="*80)
    print("정확도 계산 방법별 비교")
    print("="*80)

    for i, label in enumerate(case_labels):
        print(f"\n{label}:")
        for method in methods:
            avg = results[method]['avg'][i]
            std = results[method]['std'][i]

            if np.isnan(avg):
                print(f"  {method:20s}: nan (통계 정보 없음)")
            else:
                print(f"  {method:20s}: {avg:.4f} ± {std:.4f}")

    return results


# 사용 예시
if __name__ == "__main__":
    csv_paths = [
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
        # "results/performance_test/case24.2.1/rayleigh_lstm_MSE/performance_statistics.csv",
        # "results/performance_test/case24.2.2/rayleigh_lstm_MSE/performance_statistics.csv",
        # "results/performance_test/case24.2.3/rayleigh_lstm_MSE/performance_statistics.csv",
    ]

    # case_labels = ["no compress", "4/5", "3/5", "2/5", "1/5"]
    # case_labels = ["case 26.1", "case 26.2", "case 26.3", "case 26.4", "case 26.5", "case 26.6", "case 26.7"]
    case_labels = ["case 24.1.0", "case 24.1.1", "case 24.1.2", "case 24.1.3", "case 24.5.1", "case 24.5.2", "case 24.5.3", "case 24.3.1","case 24.3.2","case 24.3.3"]
    # case_labels = ["case 21.1","case 22.1", "case 26.5", "case 26.6", "case 26.7"]
    save_path = f"./final_comparison_plots/accuracy_methods_comparison/case{case_index.split('.')[0]}_v4/"

    # 1단계: feature_statistics.json 생성 (한 번만 실행)
    print("🔄 1단계: 피처 통계 정보 생성")
    print("   generate_feature_statistics.py를 먼저 실행하세요!\n")

    # 2단계: 정확도 계산 및 비교
    print("🔄 2단계: 정확도 계산 및 시각화")

    # 모든 방법으로 비교
    results = compare_all_methods(
        csv_paths,
        case_labels,
        save_path,
        stats_path="data/feature_statistics.json"
    )

    # 또는 특정 방법만 사용
    # final_statistic_comparison_plot_v2(
    #     csv_paths,
    #     case_labels,
    #     save_path,
    #     accuracy_method='nrmse_range',  # nrmse_range는 항상 작동
    #     stats_path="data/feature_statistics.json"
    # )
