import os
import pdb
import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from parameters.parameters import case_index


def find_case_folders(base_path: str) -> List[Path]:
    """
    casex.y 패턴의 폴더들을 찾아서 정렬된 리스트로 반환
    """
    base_path = Path(base_path)
    case_pattern = re.compile(r"^case\d+(\.\d+)+$")

    case_folders = []
    for item in base_path.iterdir():
        if item.is_dir() and case_pattern.match(item.name):
            case_folders.append(item)

    # case 번호로 정렬 (case1.0, case1.1, case2.0 순서로)
    def sort_key(folder_path):
        name = folder_path.name
        # case1.0 -> (1, 0)
        numbers = re.findall(r"\d+", name)
        return (int(numbers[0]), int(numbers[1])) if len(numbers) >= 2 else (0, 0)

    case_folders.sort(key=sort_key)
    return case_folders


def load_all_performance_data(base_path: str) -> pd.DataFrame:
    """
    모든 case 폴더에서 performance_statistics.csv를 로드하여 통합 데이터프레임 생성
    """
    case_folders = find_case_folders(base_path)
    all_data = []

    for case_folder in case_folders:
        if str(case_folder).split(".")[0].split("\\")[-1] != "case15":
            continue
        csv_midname = None

        model_case_num = str(case_folder).split(".")[1]

        if "1" in model_case_num :
            csv_midname = "deepsc"
        elif model_case_num == "2" :
            csv_midname = "lstm"
        elif model_case_num == "3" :
            csv_midname = "gru"

        csv_path = case_folder / f"rayleigh_{csv_midname}_MSE/performance_statistics.csv"

        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                df["Case"] = case_folder.name  # Case 컬럼 추가
                all_data.append(df)
                print(f"Loaded: {case_folder.name}")
            except Exception as e:
                print(f"Error loading {csv_path}: {e}")
        else:
            print(f"File not found: {csv_path}")

    if not all_data:
        raise ValueError("No valid performance data found!")

    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df


def create_comparison_plots(df: pd.DataFrame, save_path: str = None):
    """
    각 피쳐별로 MSE, MAE, RMSE 값을 비교하는 그래프 생성
    """
    # 피쳐와 메트릭 리스트 추출
    features = df["Feature"].unique()
    metrics = ["MSE", "MAE", "RMSE"]
    cases = sorted(
        df["Case"].unique(),
        key=lambda x: tuple(map(int, x.replace("case", "").split("."))),
    )

    # 전체 subplot 배치: 각 피쳐별로 3개 메트릭
    n_features = len(features)
    n_metrics = len(metrics)

    # 큰 figure 생성 (피쳐별로 3개 메트릭을 한 행에)
    fig, axes = plt.subplots(
        n_features, n_metrics, figsize=(5 * n_metrics, 4 * n_features)
    )

    # axes가 1차원인 경우 2차원으로 변환
    if n_features == 1:
        axes = axes.reshape(1, -1)
    if n_metrics == 1:
        axes = axes.reshape(-1, 1)

    # 색상 팔레트 설정
    colors = plt.cm.Set3(np.linspace(0, 1, len(cases)))

    for i, feature in enumerate(features):
        feature_data = df[df["Feature"] == feature]

        for j, metric in enumerate(metrics):
            ax = axes[i, j]

            # 각 케이스별 메트릭 값 추출
            metric_data = feature_data[feature_data["Metric"] == metric]

            if not metric_data.empty:
                # 막대 그래프
                x_pos = np.arange(len(cases))
                means = []
                stds = []

                for case in cases:
                    case_data = metric_data[metric_data["Case"] == case]
                    if not case_data.empty:
                        means.append(case_data["Mean"].iloc[0])
                        # stds.append(case_data["Std"].iloc[0])
                    else:
                        means.append(0)
                        # stds.append(0)

                bars = ax.bar(
                    x_pos,
                    means,
                    # yerr=stds,
                    capsize=5,
                    color=colors[: len(cases)],
                    alpha=0.7,
                    edgecolor="black",
                    linewidth=0.5,
                )

                # 값 표시 (평균값을 막대 위에)
                for k, (bar, mean) in enumerate(zip(bars, means)):
                    height = bar.get_height()
                    if height > 0:  # 0이 아닌 값만 표시
                        ax.text(
                            bar.get_x() + bar.get_width() / 2.0,
                            # height + stds[k],
                            height,
                            f"{mean:.4f}",
                            ha="center",
                            va="bottom",
                            fontsize=8,
                            rotation=0,
                        )

            ax.set_xlabel("Cases")
            ax.set_ylabel(f"{metric} Value")
            ax.set_title(f"{feature} - {metric}")
            ax.set_xticks(x_pos)
            ax.set_xticklabels(cases, rotation=45, ha="right")
            ax.grid(True, alpha=0.3)

            # y축 로그 스케일 (값 차이가 큰 경우)
            if (
                len(means) > 0
                and max(means) / min([m for m in means if m > 0] + [1]) > 100
            ):
                ax.set_yscale("log")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

    plt.show()


def create_heatmap_comparison(df: pd.DataFrame, save_path: str = None):
    """
    모든 케이스와 피쳐의 성능을 한눈에 비교할 수 있는 히트맵 생성
    """
    metrics = ["MSE", "MAE", "RMSE"]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for i, metric in enumerate(metrics):
        # 피벗 테이블 생성 (Case x Feature)
        metric_data = df[df["Metric"] == metric]
        pivot_table = metric_data.pivot(index="Case", columns="Feature", values="Mean")

        # 케이스 정렬
        case_order = sorted(
            pivot_table.index,
            key=lambda x: tuple(map(int, x.replace("case", "").split("."))),
        )
        pivot_table = pivot_table.reindex(case_order)

        # 히트맵 생성
        sns.heatmap(
            pivot_table,
            annot=True,
            fmt=".4f",
            cmap="YlOrRd",
            ax=axes[i],
            cbar_kws={"shrink": 0.8},
        )
        axes[i].set_title(f"{metric} Comparison Across Cases and Features")
        axes[i].set_xlabel("Features")
        axes[i].set_ylabel("Cases")

    plt.tight_layout()

    if save_path:
        heatmap_path = save_path.replace(".png", "_heatmap.png")
        plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
        print(f"Heatmap saved to: {heatmap_path}")

    plt.show()


def print_summary_statistics(df: pd.DataFrame):
    """
    요약 통계 출력
    """
    print("=" * 60)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("=" * 60)

    cases = sorted(
        df["Case"].unique(),
        key=lambda x: tuple(map(int, x.replace("case", "").split("."))),
    )
    print(f"Total Cases Found: {len(cases)}")
    print(f"Cases: {', '.join(cases)}")
    print(f"Features: {', '.join(df['Feature'].unique())}")
    print(f"Metrics: {', '.join(df['Metric'].unique())}")

    print("\n" + "=" * 60)
    print("BEST PERFORMANCE (Lowest Values)")
    print("=" * 60)

    for metric in ["MSE", "MAE", "RMSE"]:
        print(f"\n--- {metric} ---")
        metric_data = df[df["Metric"] == metric]
        for feature in df["Feature"].unique():
            feature_data = metric_data[metric_data["Feature"] == feature]
            if not feature_data.empty:
                best_case = feature_data.loc[feature_data["Mean"].idxmin()]
                print(
                    f"{feature:20s}: {best_case['Case']:10s} (Mean: {best_case['Mean']:.6f})"
                )


def main():
    """
    메인 실행 함수
    """
    # 경로 설정 (현재 디렉토리에서 case 폴더들을 찾음)
    base_path = "./results/performance_test/"  # 또는 특정 경로 지정: "/path/to/your/cases"

    try:
        # 모든 성능 데이터 로드
        print("Loading performance data from all cases...")
        df = load_all_performance_data(base_path)

        # df에서 time 제거
        # df = df[df.Feature != "Time"]

        # 요약 통계 출력
        print_summary_statistics(df)

        # 비교 그래프 생성
        print("\nGenerating comparison plots...")
        create_comparison_plots(
            df, save_path=f"./analysis/case_{case_index}/performance_comparison.png"
        )

        # 히트맵 생성
        print("Generating heatmap...")
        create_heatmap_comparison(
            df, save_path=f"./analysis/case_{case_index}/performance_comparison.png"
        )

        print("\nAnalysis completed successfully!")

    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
