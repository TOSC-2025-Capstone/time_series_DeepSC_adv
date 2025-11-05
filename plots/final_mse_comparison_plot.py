import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os

# --- 1. 사용자 설정 영역 ---

# 비교할 지표를 선택함 ('MSE', 'MAE', 'RMSE' 중 하나).
SELECTED_METRIC = 'MSE'

date = "251105"  # 그래프 저장용 날짜 디렉토리 이름

# 모델별 인덱스와 기본 경로 템플릿을 설정함.
MODEL_CONFIG = {
    'Inverted-Transformer': 1,
    # 'LSTM': 2,
    # 'GRU': 3,
    # 'Transformer': 4,
}
# BASE_PATH_TEMPLATE = "results/performance_test/case{case_idx}.{model_idx}.{snr_idx}/{sub_dir}/performance_statistics.csv"
BASE_PATH_TEMPLATE = "results/performance_test/case60.{model_idx}.{snr_idx}/{sub_dir}/performance_statistics.csv"

# 비교할 SNR 리스트를 정의함.
SNRS = [3, 6, 9, 12, 15, 18, 21]

# SNR 값과 경로의 snr_idx를 매핑함.
SNR_TO_INDEX_MAP = {
    3: 2,
    6: 3,
    9: 4,
    12: 5,
    15: 6,
    18: 7,
    21: 8,
}

CASE_TO_INDEX_MAP = {
    60: 1,
    61: 2,
    62: 3,
    63: 4,
    64: 5,
    65: 6,
    66: 7,
    67: 8,
    68: 9,
    69: 10,
}

# 각 모델의 하위 디렉토리 이름 (경로에 따라 수정 필요함).
SUB_DIR_TEMPLATE = "rayleigh_{model_name_lower}_MSE"


# --- 2. 파일 경로 자동 생성 ---
def generate_file_paths(snrs, model_config, snr_map):
    """설정값을 기반으로 파일 경로 딕셔너리를 동적으로 생성함."""
    paths = {}
    for snr_val in snrs:
        if snr_val not in snr_map:
            print(f"Warning: SNR value {snr_val} is not in the SNR_TO_INDEX_MAP. Skipping.")
            continue

        snr_idx = snr_map[snr_val] # 맵에서 인덱스를 조회함.

        for model_name, model_idx in model_config.items():
            key = f"{model_name}_{snr_val}db"

            # 하위 디렉토리 이름 결정 (Inverted-Transformer와 Transformer는 'deepsc' 사용을 가정함).
            if 'Transformer' in model_name:
                sub_dir_model_name = 'deepsc'
            else:
                sub_dir_model_name = model_name.lower()

            sub_dir = SUB_DIR_TEMPLATE.format(model_name_lower=sub_dir_model_name)

            path = BASE_PATH_TEMPLATE.format(
                model_idx=model_idx,
                snr_idx=snr_idx,
                sub_dir=sub_dir
            )
            paths[key] = path
    return paths

FILE_PATHS = generate_file_paths(SNRS, MODEL_CONFIG, SNR_TO_INDEX_MAP)

# --- 3. 데이터 처리 및 플로팅 ---
PLOT_VALUE = 'Mean' # CSV 파일에서 사용할 값 (Mean).

def load_and_preprocess_data(file_paths, selected_metric):
    """지정된 경로에서 데이터를 로드하고 요청된 지표로 통합함."""
    results = []
    for key, path in file_paths.items():
        try:
            model_name = key.split('_')[0]
            snr = int(key.split('_')[1].replace('db', ''))
            df_stats = pd.read_csv(path)
            metric_rows = df_stats[df_stats['Metric'] == selected_metric]

            if metric_rows.empty:
                print(f"Warning: Metric '{selected_metric}' not found in {path}")
                continue

            avg_value = metric_rows[PLOT_VALUE].mean()
            results.append({
                'SNR': snr,
                'Model': model_name,
                selected_metric: avg_value
            })
        except FileNotFoundError:
            print(f"Info: File not found at {path}. Skipping.")
        except Exception as e:
            print(f"Error processing {path}: {e}. Skipping.")
    return pd.DataFrame(results)

# --- plot_comparison 함수 (수정됨) ---
def plot_comparison(df, selected_metric, model_names_ordered, baseline_model='LSTM'):
    """
    통합된 데이터를 사용하여 모델별 상대적 성능 꺾은선 그래프를 생성함.
    지정된 baseline_model 대비 비율로 값을 정규화함.
    """
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 7)) # Figure 크기 조정

    # --- 데이터 정규화 로직 추가 ---
    df_normalized = df.copy()
    baseline_values = df_normalized[df_normalized['Model'] == baseline_model].set_index('SNR')[selected_metric]

    # 각 모델, 각 SNR에 대해 baseline 값으로 나눔
    df_normalized[f'Relative_{selected_metric}'] = df_normalized.apply(
        lambda row: row[selected_metric] / baseline_values.get(row['SNR'], np.nan)
                    if row['SNR'] in baseline_values and not np.isnan(baseline_values.get(row['SNR'])) and baseline_values.get(row['SNR']) != 0
                    else np.nan,
        axis=1
    )
    # -----------------------------

    # 논문용 회색조 스타일 및 마커/선 스타일을 정의함.
    cmap = cm.get_cmap('Greys')
    grayscale_colors = [cmap(i) for i in np.linspace(0.8, 0.2, len(model_names_ordered))]
    markers = ['o', 's', '^', 'D']
    linestyles = ['-', '--', ':', '-.']

    style_map = {
        model: {
            'color': grayscale_colors[i],
            'marker': markers[i % len(markers)], # 마커 순환
            'linestyle': linestyles[i % len(linestyles)] # 라인스타일 순환
        }
        for i, model in enumerate(model_names_ordered)
    }

    # 꺾은선 그래프를 그림 (정규화된 값 사용).
    plot_metric_col = f'Relative_{selected_metric}' # 사용할 컬럼 이름
    for model in model_names_ordered:
        subset = df_normalized[df_normalized['Model'] == model].sort_values(by='SNR')
        if subset.empty or subset[plot_metric_col].isnull().all(): # 데이터가 없거나 모두 NaN이면 건너뜀
             print(f"Skipping plot for {model} due to missing or all NaN relative values.")
             continue
        style = style_map.get(model)

        ax.plot(
            subset['SNR'],
            subset[plot_metric_col], # 정규화된 값 사용
            color=style['color'],
            marker=style['marker'],
            linestyle=style['linestyle'],
            label=model,
            linewidth=2.5, # 선 굵기 증가
            markersize=8.0 # 마커 크기 증가
        )

    # 그래프 제목 및 축 레이블을 설정함 (상대적 성능임을 명시).
    ax.set_title(f'Relative Model Performance vs. SNR (vs. {baseline_model}, Metric: Avg {selected_metric})', fontsize=16, fontweight='bold')
    ax.set_xlabel('SNR (dB)', fontsize=14)
    # Y축 레이블 수정
    ax.set_ylabel(f'Relative Avg {selected_metric} (Ratio to {baseline_model})', fontsize=14)

    # x축 눈금을 설정함.
    ax.set_xticks(SNRS)
    # y축 눈금 라벨 포맷 지정 (소수점 표시)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))


    # 기준선 (1.0) 추가
    ax.axhline(1.0, color='black', linestyle='--', linewidth=1.5, label=f'{baseline_model} Baseline (Ratio=1.0)')

    # 범례 및 격자를 추가함.
    ax.legend(title='Model', loc='best', frameon=True, fontsize=12) # 범례 폰트 크기 조정
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7) # 격자 스타일 변경

    # Y축 범위 조정 (선택 사항, 데이터 분포에 따라 조절)
    # ax.set_ylim(bottom=0) # 최소값을 0으로 설정하거나
    # ax.set_ylim(0, df_normalized[plot_metric_col].max() * 1.1) # 최대값에 약간의 여유를 둠

    plt.tight_layout()

    # 'plots' 디렉토리를 생성함.
    if not os.path.exists('plots'):
        os.makedirs('plots')

    # 그래프를 저장함 (파일명에 'relative' 추가).
    os.makedirs(f'plots/{date}/', exist_ok=True)
    filename = f'plots/{date}/model_comparison_relative_{selected_metric}.png'
    plt.savefig(filename, dpi=300)
    print(f"\nRelative performance graph saved as '{filename}'")
    print("---------------------------------------------------------")
    print("\nProcessed Data Summary (with Relative Values):")
    # 보기 좋게 정렬하여 출력함.
    print(df_normalized.sort_values(['SNR', 'Model']).reset_index(drop=True))


# --- 4. 메인 실행 ---
if __name__ == "__main__":
    print("--- Generating file paths based on configuration ---")
    print(f"{len(FILE_PATHS)} paths generated.")
    print("---------------------------------------------------------")

    df_results = load_and_preprocess_data(FILE_PATHS, SELECTED_METRIC)

    if not df_results.empty:
        # 모델 순서 지정 (LSTM이 맨 앞에 오도록)
        # ordered_models = ['LSTM'] + [m for m in MODEL_CONFIG.keys() if m != 'LSTM']
        # plot_comparison(df_results, SELECTED_METRIC, ordered_models, baseline_model='LSTM')
        # ordered_models = ['LSTM'] + [m for m in MODEL_CONFIG.keys() if m != 'LSTM']
        plot_comparison(df_results, SELECTED_METRIC, MODEL_CONFIG.keys(), baseline_model='Inverted-Transformer')
    else:
        print("No data available to generate a plot.")
