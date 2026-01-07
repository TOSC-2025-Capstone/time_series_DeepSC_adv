import os
import pdb
import textwrap # 자동 줄바꿈을 위해 추가

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from parameters.parameters import case_index

# === 스타일 설정 강화 ===
# (스타일 설정 코드는 이전과 동일하게 유지)
# 한글 폰트 설정 - 영어로 대체
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# Seaborn 스타일 및 추가 설정 적용
plt.style.use("seaborn-v0_8-whitegrid") # 그리드 추가
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "figure.titlesize": 22, # 전체 그림 제목 크기 조정
        "axes.titlesize": 18,   # ★ 개별 플롯 제목 크기 조정
        "axes.labelsize": 18,   # ★ 축 레이블 크기 조정
        "xtick.labelsize": 16,  # X축 틱 라벨 크기
        "ytick.labelsize": 16,  # Y축 틱 라벨 크기
        "legend.fontsize": 16,  # ★ 범례 크기 조정
        "grid.color": "grey",   # 그리드 색상
        "grid.linestyle": ":",  # 그리드 스타일
        "grid.linewidth": 0.5,  # 그리드 두께
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "black", # 축 테두리 색상
        "legend.frameon": True,   # 범례 테두리
        "legend.edgecolor": "lightgray",
        "mathtext.fontset": "dejavusans",
        "mathtext.default": "regular",
        "text.usetex": False,
        "font.style": "normal",
    }
)
# 전문적인 색상 팔레트 (Seaborn colorblind 확장) + 검정
# publication_palette = [
#     "#0173B2", "#DE8F05", "#029E73", "#D55E00",
#     "#CC78BC", "#CA9161", "#FBAFE4", "#949494",
#     "#ECE133", "#56B4E9", "#000000" # 검정 추가
# ]

publication_palette = [
    # "#000000",  # Pure Black (검정)
    # "#2C2C2C",  # Very Dark Gray (매우 짙은 회색)
    # "#4A4A4A",  # Dark Gray (짙은 회색)
    "#6B6B6B",  # Medium-Dark Gray (중간-짙은 회색)
    "#8B8B8B",  # Medium Gray (중간 회색)
    "#ABABAB",  # Medium-Light Gray (중간-밝은 회색)
    "#1A1A1A",  # Almost Black (거의 검정)
    "#5A5A5A",  # Dark-Medium Gray (짙은-중간 회색)
]

# === 1. 막대 차트 함수 (상대 성능 비교) ===
# (plot_metric_comparison_bars 함수는 이전과 동일)
def plot_metric_comparison_bars(csv_paths, case_labels=None, save_path=None, metric_type=None, wrap_width=15):
    """지표(MSE, MAE, RMSE 등)에 대해 첫 번째 케이스 대비 상대적 성능을 막대 차트로 시각화"""
    # 데이터 읽기
    dfs = [pd.read_csv(path) for path in csv_paths]
    if case_labels is None:
        case_labels = [f"Case {i+1}" for i in range(len(dfs))]

    base_label = case_labels[0] # 비교 기준 모델 라벨

    # Feature 순서 및 라벨 정의 (Time 제외, 줄바꿈 적용)
    features = [f for f in dfs[0]["Feature"].unique() if f != "Time"]
    features_label_raw = [f.replace("_", " ") for f in features]
    features_label = [textwrap.fill(label, width=wrap_width) for label in features_label_raw]
    features.append("Average")
    features_label.append("Average")

    # 지표별 플롯 생성
    metric = metric_type
    fig, ax = plt.subplots(figsize=(18, 7)) # 크기 조정
    bar_width = 0.15 # 막대 너비 조정
    num_cases = len(dfs)
    x = np.arange(len(features)) # X축 위치 (피처 중앙)

    base_values = None # 기준 모델 값 저장

    for i, df in enumerate(dfs):
        # 각 feature에 대해 metric 값 추출
        values = []
        for feature in features[:-1]: # Average 제외
            row = df[(df["Feature"] == feature) & (df["Metric"] == metric)]
            val = row["Mean"].values[0] if not row.empty else np.nan
            values.append(val)

        # 기준 모델 값 기준으로 비율 계산
        if i == 0:
            base_values = values # 기준 값 저장
            normalized_values = [1.0] * len(values) # 기준 모델 자체는 1
        else:
            normalized_values = []
            for v, bv in zip(values, base_values):
                # 기준값이 0이거나 NaN이면 비율 계산 불가
                if bv is None or np.isnan(bv) or bv == 0:
                    normalized_values.append(np.nan)
                elif v is None or np.isnan(v):
                     normalized_values.append(np.nan)
                else:
                    normalized_values.append(v / bv)

        # Average 계산 및 추가
        avg_relative = np.nanmean(normalized_values)
        normalized_values.append(avg_relative)

        # 막대 위치 계산 (피처 중앙 기준 offset)
        offset = (i - (num_cases - 1) / 2) * bar_width
        color = publication_palette[i % len(publication_palette)]

        # 막대 그리기
        bars = ax.bar(
            x + offset,
            normalized_values,
            width=bar_width,
            label=case_labels[i],
            color=color,
            edgecolor="black", # 테두리 추가
            linewidth=0.5,
        )
        # Average 막대에 해치(빗금) 추가
        bars[-1].set_hatch("///")

        # 막대 위에 값 표시 (소수점 둘째 자리)
        ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=10) # bar_label 사용

    # 그래프 꾸미기
    ax.set_xticks(x)
    # ax.set_xticklabels(features_label, rotation=0, ha="center") # 회전 제거, 중앙 정렬
    wrapped_labels = [textwrap.fill(label, width=12) for label in features_label]
    ax.set_xticklabels(wrapped_labels)
    ax.set_ylabel(f"Relative {metric}\n(Ratio to {base_label})")
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, label=f"{base_label} Baseline (1.0)") # 기준선
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f')) # Y축 포맷
    # ax.set_ylim(bottom=0) # Y축 최소값을 0으로 설정 (선택 사항)

    # 범례 위치 조정 (그림 위쪽 중앙)
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=min(num_cases + 1, 4), frameon=True)

    plt.suptitle(f"Feature-wise {metric} Comparison (Relative to {base_label})", y=1.02) # 제목 위치 조정
    plt.tight_layout(rect=[0, 0, 1, 0.92]) # 범례 공간 확보

    # 저장 또는 출력
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, f"{metric}_relative_bar_comparison.png")
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"[✓] 저장 완료 (상대 성능 막대): {save_file}")
    else:
        plt.show()
    plt.close(fig)

# === 2. 원본 vs 복원 비교 플롯 함수 (서브플롯) ===
# (plot_reconstruction_comparison 함수는 이전과 동일)
def plot_reconstruction_comparison(
    original_path: str, model_paths: dict, feature_names: list,
    save_path: str, filename: str
):
    """원본 시계열과 여러 모델의 복원 결과를 비교하여 플롯 (하나의 그림에 서브플롯으로)"""
    # 데이터 로드
    df_original = pd.read_csv(os.path.join(original_path, filename))
    model_dfs = {}
    for i, (name, path) in enumerate(model_paths.items()):
        try:
            df_model = pd.read_csv(os.path.join(path, filename.replace(".csv", "_reconstructed.csv")))
            model_dfs[name] = df_model
        except FileNotFoundError:
            print(f"Warning: Reconstructed file not found for model '{name}' at path '{path}'. Skipping.")
            continue # 파일 없으면 건너뜀

    if not model_dfs:
        print("Error: No reconstructed data loaded. Aborting plot.")
        return

    num_features = len(feature_names)
    ncols = 3
    nrows = (num_features + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), sharex=True)
    axes = axes.flatten() # 축 배열을 1차원으로 만듦

    # 라인 스타일 및 마커 정의 (모델별로 고정)
    linestyles = ['--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 5)), (0, (1, 1))]
    colors = publication_palette[1:] # 첫 번째 색상(파랑)은 다른 용도로 사용 가능

    # 각 피처별로 subplot 그리기
    for i, feature in enumerate(feature_names):
        ax = axes[i]

        # 원본 데이터 플롯 (굵은 검정색)
        ax.plot(df_original.index, df_original[feature], label="Original", color="black", linewidth=2.5, zorder=5)

        # 모델별 복원 데이터 플롯
        model_idx = 0
        for name, df_model in model_dfs.items():
            if feature in df_model.columns:
                style = linestyles[model_idx % len(linestyles)]
                color = colors[model_idx % len(colors)]
                ax.plot(df_model.index, df_model[feature], label=name, color=color, linestyle=style, linewidth=1.5, alpha=0.9)
                model_idx += 1
            else:
                 print(f"Warning: Feature '{feature}' not found in reconstructed data for model '{name}'.")


        ax.set_title(feature.replace("_", " "), fontsize=16) # Subplot 제목 크기 조정
        ax.set_ylabel(feature.replace("_", " "))
        if i >= num_features - ncols: # 마지막 행에만 X축 레이블 표시
             ax.set_xlabel("Timestep")
        ax.tick_params(axis='both', which='major', labelsize=12) # 틱 라벨 크기 조정

    # 사용하지 않는 subplot 숨기기
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # 공유 범례 생성 (그림 하단 중앙)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=min(len(model_dfs) + 1, 5), frameon=True)

    plt.suptitle("Original vs. Reconstructed Time Series", y=1.0) # 제목 위치 조정
    plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # 범례 공간 확보

    # 저장
    os.makedirs(save_path, exist_ok=True)
    output_file = os.path.join(save_path, f"comparison_subplots_{filename.replace('.csv', '')}.png") # 파일명 변경
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[✓] 저장 완료 (원본-복원 서브플롯): {output_file}")
    plt.close(fig)

# === 3. 잔차 제곱 플롯 함수 ===
# (plot_residual_comparison 함수는 이전과 동일)
def plot_residual_comparison(
    original_path: str, model_paths: dict, feature_names: list,
    save_path: str, filename: str
):
    """원본과 모델 복원 결과 간의 잔차 제곱(Squared Error)을 플롯"""
    # 데이터 로드
    df_original = pd.read_csv(os.path.join(original_path, filename))
    model_dfs = {}
    for i, (name, path) in enumerate(model_paths.items()):
        try:
            df_model = pd.read_csv(os.path.join(path, filename.replace(".csv", "_reconstructed.csv")))
            model_dfs[name] = df_model
        except FileNotFoundError:
            print(f"Warning: Reconstructed file not found for model '{name}' at path '{path}'. Skipping.")
            continue

    if not model_dfs:
        print("Error: No reconstructed data loaded. Aborting plot.")
        return

    num_features = len(feature_names)
    ncols = 3
    nrows = (num_features + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), sharex=True)
    axes = axes.flatten()

    # 스타일 정의 (reconstruction과 동일하게 유지)
    linestyles = ['--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 5)), (0, (1, 1))]
    colors = publication_palette[1:]

    # 각 피처별로 subplot 그리기
    for i, feature in enumerate(feature_names):
        ax = axes[i]
        max_error = 0 # Y축 스케일 조정을 위함

        # 모델별 잔차 제곱 플롯
        model_idx = 0
        for name, df_model in model_dfs.items():
             if feature in df_model.columns and feature in df_original.columns:
                residual_sq = (df_original[feature] - df_model[feature]) ** 2
                style = linestyles[model_idx % len(linestyles)]
                color = colors[model_idx % len(colors)]
                ax.plot(df_original.index, residual_sq, label=name, color=color, linestyle=style, linewidth=1.0) # 선 두께 조정
                max_error = max(max_error, residual_sq.max())
                model_idx += 1
             else:
                print(f"Warning: Feature '{feature}' not found in original or reconstructed data for model '{name}'. Skipping residual plot.")


        ax.set_title(f"{feature.replace('_', ' ')} - Squared Error", fontsize=16)
        ax.set_ylabel("Squared Error")
        if i >= num_features - ncols:
             ax.set_xlabel("Timestep")

        # Y축 스케일 조정 (0부터 시작하도록)
        if max_error > 0:
            ax.set_ylim(bottom=-0.05 * max_error, top=max_error * 1.1) # 약간의 여백 추가
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) # 과학적 표기법 사용
        ax.tick_params(axis='both', which='major', labelsize=12)

    # 사용하지 않는 subplot 숨기기
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # 공유 범례 생성
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=min(len(model_dfs), 5), frameon=True)

    plt.suptitle("Squared Error (Original vs. Reconstructed)", y=1.0)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])

    # 저장
    os.makedirs(save_path, exist_ok=True)
    output_file = os.path.join(save_path, f"residual_sq_{filename.replace('.csv', '')}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[✓] 저장 완료 (잔차 제곱 비교): {output_file}")
    plt.close(fig)

# ===  4. 개별 피처 비교 플롯 함수 ===
def plot_individual_feature_comparison(
    original_path: str, model_paths: dict, feature_names: list,
    save_path: str, filename: str
):
    """각 피처별로 원본과 모델 복원 결과를 비교하는 개별 그림 생성"""
    # 데이터 로드 (한 번만 로드하여 효율성 증대)
    try:
        df_original = pd.read_csv(os.path.join(original_path, filename))
    except FileNotFoundError:
        print(f"Error: Original file not found at '{os.path.join(original_path, filename)}'. Aborting.")
        return

    model_dfs = {}
    for i, (name, path) in enumerate(model_paths.items()):
        try:
            df_model = pd.read_csv(os.path.join(path, filename.replace(".csv", "_reconstructed.csv")))
            model_dfs[name] = df_model
        except FileNotFoundError:
            print(f"Warning: Reconstructed file not found for model '{name}' at path '{path}'. Skipping.")
            # 해당 모델은 이 함수에서 제외됨

    if not model_dfs:
        print("Warning: No reconstructed data loaded for individual plots.")
        # 원본 데이터만이라도 그릴 수 있으나, 비교 의미가 없으므로 종료
        # return

    # 스타일 정의 (다른 플롯과 일관성 유지)
    linestyles = ['--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 5)), (0, (1, 1))]
    colors = publication_palette[1:] # publication_palette 사용

    # 각 피처별로 개별 그림 생성
    for feature in feature_names:
        if feature not in df_original.columns:
            print(f"Warning: Feature '{feature}' not found in original data. Skipping plot.")
            continue

        fig, ax = plt.subplots(figsize=(10, 6)) # 개별 그림 크기

        # 원본 데이터 플롯
        ax.plot(df_original.index, df_original[feature], label="Original", color="black", linewidth=2.5, zorder=5)

        # 모델별 복원 데이터 플롯
        model_idx = 0
        for name, df_model in model_dfs.items():
            if feature in df_model.columns:
                style = linestyles[model_idx % len(linestyles)]
                color = colors[model_idx % len(colors)]
                ax.plot(df_model.index, df_model[feature], label=name, color=color, linestyle=style, linewidth=1.5, alpha=0.9)
                model_idx += 1
            else:
                print(f"Warning: Feature '{feature}' not found in reconstructed data for model '{name}' (individual plot).")


        # 그래프 꾸미기 (개별 플롯에 맞게 조정)
        feature_title = feature.replace("_", " ")
        ax.set_title(f"Comparison for {feature_title}")
        ax.set_xlabel("Timestep")
        ax.set_ylabel(feature_title)
        ax.legend(loc='best') # 범례 위치 자동 조정
        ax.tick_params(axis='both', which='major', labelsize=14) # 개별 플롯 틱 라벨 크기

        plt.tight_layout()

        # 저장 (피처별 하위 디렉토리 생성)
        feature_save_path = os.path.join(save_path, "individual_features")
        os.makedirs(feature_save_path, exist_ok=True)
        output_file = os.path.join(feature_save_path, f"comparison_{feature}_{filename.replace('.csv', '')}.png")
        try:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"[✓] 저장 완료 (개별 피처: {feature}): {output_file}")
        except Exception as e:
            print(f"Error saving file {output_file}: {e}")

        plt.close(fig) # 다음 피처를 위해 현재 그림 닫기


# === 5. 평균 지표 계산 함수 (함수명 수정 및 기능 동일) ===
def print_avg_metric_excluding_time(csv_paths, labels, metric_type="MSE"):
    """각 CSV에서 특정 지표(Metric)의 평균('Mean')을 계산 (Time 피처 제외)"""
    print(f"\n--- Average {metric_type} (excluding 'Time') ---")
    results = {}
    for path, label in zip(csv_paths, labels):
        try:
            df = pd.read_csv(path)
            # 지정된 지표 및 Time 제외 필터링
            df_filtered = df[(df["Metric"] == metric_type) & (df["Feature"] != "Time")]
            if df_filtered.empty:
                print(f"{label}: No '{metric_type}' data found (excluding Time).")
                avg = np.nan
            else:
                avg = df_filtered["Mean"].mean()
                print(f"{label}: avg {metric_type} = {avg:.6f}")
            results[label] = avg
        except FileNotFoundError:
            print(f"Error: File not found at {path}. Skipping calculation for {label}.")
            results[label] = np.nan
        except Exception as e:
            print(f"Error processing file {path} for {label}: {e}")
            results[label] = np.nan
    return results


# === 실행 블록 ===
if __name__ == "__main__":
    # --- 설정 ---
    prefix = "./results/performance_test"
    case_id_lstm = "case10045.2.8"
    case_id_gru = "case10045.3.8"
    case_id_transformer = "case10050.4.8"
    case_id_itransformer = "case10045.1.8"
    channel = "AWGN"
    metric_loss = "MSE" # 학습 시 사용된 손실 함수 (경로명에 사용됨)

    # filename_to_compare = "01197.csv" # 비교할 특정 사이클 파일명
    filename_to_compare = "02531.csv" # 비교할 특정 사이클 파일명
    snr_condition = "21db" # 결과 저장 경로에 사용할 조건명
    date = "251227"

    # 모델별 통계 CSV 경로
    csv_paths = [
        f"{prefix}/{case_id_lstm}/{channel}_{'lstm'}_{metric_loss}/performance_statistics.csv",
        f"{prefix}/{case_id_gru}/{channel}_{'gru'}_{metric_loss}/performance_statistics.csv",
        f"{prefix}/{case_id_transformer}/{channel}_{'deepsc'}_{metric_loss}/performance_statistics.csv",
        f"{prefix}/{case_id_itransformer}/{channel}_{'deepsc'}_{metric_loss}/performance_statistics.csv",
    ]
    # 모델 라벨 (순서 중요!)
    case_labels = ["LSTM", "GRU", "Transformer", "InvertedTransformer"]

    # 비교할 원본 및 복원 데이터 경로
    original_data_path = "./cycle_preprocess/csv/outlier_cut/threshold_7/cycle_len_512"
    model_reconstruction_paths = {
        # 라벨 순서와 key 순서 일치시키기 (Dictionary는 순서 보장 안될 수 있으나 가급적)
        "LSTM": f"./reconstruction/{case_id_lstm}/reconstructed_{channel.lower()}_lstm_{metric_loss}",
        "GRU": f"./reconstruction/{case_id_gru}/reconstructed_{channel.lower()}_gru_{metric_loss}",
        "Transformer": f"./reconstruction/{case_id_transformer}/reconstructed_{channel.lower()}_deepsc_{metric_loss}",
        "InvertedTransformer": f"./reconstruction/{case_id_itransformer}/reconstructed_{channel.lower()}_deepsc_{metric_loss}",
    }

    # 분석할 피처 목록
    feature_names_to_plot = [
        "Voltage_measured", "Current_measured", "Temperature_measured",
        "Current_load", "Voltage_load", # "Time", # Time은 보통 제외
    ]

    # 저장 경로 설정
    save_path_base = f"./final_comparison_plots/{date}_10050.4/case{str(case_index).split('.')[0]}/{filename_to_compare.replace('.csv','')}/{snr_condition}/"

    # --- 실행 ---
    print("[📊] Starting Plot Generation...")

    # 1. 상대 성능 막대 차트 생성 (MSE 기준)
    print("\n--- Plotting: Relative Metric Bars (MSE) ---")
    plot_metric_comparison_bars(
        csv_paths, case_labels, save_path=save_path_base, metric_type="MSE", wrap_width=10
    )
    # (필요시 MAE, RMSE 추가)

    # 2. 원본 vs 복원 시계열 비교 플롯 생성 (서브플롯 버전)
    print("\n--- Plotting: Reconstruction Comparison (Subplots) ---")
    plot_reconstruction_comparison(
        original_path=original_data_path,
        model_paths=model_reconstruction_paths,
        feature_names=feature_names_to_plot,
        save_path=save_path_base,
        filename=filename_to_compare,
    )

    # 3. 잔차 제곱 비교 플롯 생성 (서브플롯 버전)
    print("\n--- Plotting: Residual Squared Comparison (Subplots) ---")
    plot_residual_comparison(
        original_path=original_data_path,
        model_paths=model_reconstruction_paths,
        feature_names=feature_names_to_plot,
        save_path=save_path_base,
        filename=filename_to_compare,
    )

    # 🚀 4. 개별 피처 비교 플롯 생성 (신규 추가된 함수 호출)
    print("\n--- Plotting: Individual Feature Comparisons ---")
    plot_individual_feature_comparison(
        original_path=original_data_path,
        model_paths=model_reconstruction_paths,
        feature_names=feature_names_to_plot,
        save_path=save_path_base, # 기본 저장 경로 전달
        filename=filename_to_compare,
    )

    # 5. 평균 지표 계산 및 출력 (MSE)
    avg_results_mse = print_avg_metric_excluding_time(csv_paths, case_labels, metric_type="MSE")
    # (필요시 MAE, RMSE 추가)

    print("\n[🎉] All plots and calculations completed.")
