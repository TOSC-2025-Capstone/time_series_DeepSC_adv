import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os

# --- 1. 사용자 설정 영역 ---

# 비교할 지표를 선택함 ('MSE', 'MAE', 'RMSE' 중 하나).
SELECTED_METRIC = 'MSE'

# 모델별 인덱스와 기본 경로 템플릿을 설정함.
MODEL_CONFIG = {
    'Inverted-Transformer': 1,
    'LSTM': 2,
    'GRU': 3,
    'Transformer': 4,
}
BASE_PATH_TEMPLATE = "results/performance_test/case46.{model_idx}.{snr_idx}/{sub_dir}/performance_statistics.csv"

# 비교할 SNR 리스트를 정의함.
SNRS = [3, 6, 9, 12, 15, 18, 21]

# SNR 값과 경로의 snr_idx를 매핑함.
SNR_TO_INDEX_MAP = {
    3: 5,
    6: 6,
    9: 7,
    12: 8,
    15: 3,
    18: 9,
    21: 10,
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

def plot_comparison(df, selected_metric, model_names_ordered):
    """통합된 데이터를 사용하여 꺾은선 그래프를 생성함."""
    plt.style.use('default') # 기본 스타일로 초기화함.
    fig, ax = plt.subplots(figsize=(10, 6))

    # 논문용 회색조 스타일 및 마커/선 스타일을 정의함.
    cmap = cm.get_cmap('Greys')
    grayscale_colors = [cmap(i) for i in np.linspace(0.8, 0.2, len(model_names_ordered))]
    markers = ['o', 's', '^', 'D']
    linestyles = ['-', '--', ':', '-.']

    style_map = {
        model: {
            'color': grayscale_colors[i],
            'marker': markers[i],
            'linestyle': linestyles[i]
        }
        for i, model in enumerate(model_names_ordered)
    }

    # 꺾은선 그래프를 그림.
    for model in model_names_ordered:
        subset = df[df['Model'] == model].sort_values(by='SNR')
        if subset.empty:
            continue
        style = style_map.get(model)

        ax.plot(
            subset['SNR'],
            subset[selected_metric],
            color=style['color'],
            marker=style['marker'],
            linestyle=style['linestyle'],
            label=model,
            linewidth=2.0,
            markersize=7.0
        )

    # 그래프 제목 및 축 레이블을 설정함.
    ax.set_title(f'Model Performance vs. SNR (Metric: Average {selected_metric})', fontsize=16, fontweight='bold')
    ax.set_xlabel('SNR (dB)', fontsize=14)
    ax.set_ylabel(f'Average {selected_metric} Across All Features', fontsize=14)

    # x축 눈금을 설정함.
    ax.set_xticks(SNRS)

    # 범례 및 격자를 추가함.
    ax.legend(title='Model', loc='best', frameon=True)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # y축 로그 스케일 적용 (필요 시 주석을 해제함).
    # ax.set_yscale('log')

    plt.tight_layout()

    # 'plots' 디렉토리를 생성함.
    if not os.path.exists('plots'):
        os.makedirs('plots')

    # 그래프를 저장함.
    filename = f'plots/model_comparison_{selected_metric}.png'
    plt.savefig(filename, dpi=300)
    print(f"\nGraph saved as '{filename}'")
    print("---------------------------------------------------------")
    print("\nProcessed Data Summary:")
    # 보기 좋게 정렬하여 출력함.
    print(df.sort_values(['SNR', 'Model']).reset_index(drop=True))


# --- 4. 메인 실행 ---
if __name__ == "__main__":
    print("--- Generating file paths based on configuration ---")
    print(f"{len(FILE_PATHS)} paths generated.")
    print("---------------------------------------------------------")

    df_results = load_and_preprocess_data(FILE_PATHS, SELECTED_METRIC)

    if df_results.empty:
        print("\n[경고] 유효한 파일 경로 데이터를 찾을 수 없음. 시연을 위해 더미 데이터를 사용함.")

        # 확장된 SNR 리스트에 맞는 더미 데이터를 생성함.
        dummy_data_list = []
        model_names_cycle = list(MODEL_CONFIG.keys())
        for snr in SNRS:
            for model in model_names_cycle:
                # SNR이 증가할수록 에러가 감소하는 경향을 보이는 더미 값을 생성함.
                base_error = 1.0 / (snr + 1)
                model_factor = {'LSTM': 1.2, 'GRU': 1.5, 'Transformer': 1.0, 'Inverted-Transformer': 0.8}.get(model, 1)
                error_value = base_error * model_factor * (1 + (np.random.rand() - 0.5) * 0.2)
                dummy_data_list.append({'SNR': snr, 'Model': model, SELECTED_METRIC: error_value})

        df_results = pd.DataFrame(dummy_data_list)

    if not df_results.empty:
        plot_comparison(df_results, SELECTED_METRIC, list(MODEL_CONFIG.keys()))
    else:
        print("No data available to generate a plot.")
