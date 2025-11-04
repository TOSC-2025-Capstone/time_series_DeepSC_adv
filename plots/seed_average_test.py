import pandas as pd
import os
import glob
import pdb
import io
import matplotlib.pyplot as plt
import re
import numpy as np

# --- 설정 변수 ---
MODEL_INDICES = [1, 2] # 1: InvertedTransformer, 2: LSTM (예시)
MODEL_NAMES = {1: "deepsc", 2: "lstm"} # 인덱스에 따른 실제 모델 이름
SNR_INDICES = list(range(2, 9)) # Z=1 (3dB) ~ Z=7 (21dB). (요청에는 2~8이라고 했으나, 3dB부터 7개이므로 1~7로 가정. 필요시 수정)
CASE_RANGE = range(60, 70) # X=1부터 X=10까지

# 경로 템플릿
BASE_PATH_TEMPLATE = "results/performance_test/case{case_num}.{model_idx}.{snr_idx}/{sub_dir}/performance_statistics.csv"
SUB_DIR_TEMPLATE = "rayleigh_{model_name_lower}_MSE"
# ------------------


def calculate_average_performance():
    """
    10개 케이스(Seed)의 성능 통계를 읽어와 Feature 및 Metric 별 평균을 계산합니다.
    """

    # 결과를 저장할 최종 딕셔너리
    final_averages = {}

    for model_idx in MODEL_INDICES:
        model_name_lower = MODEL_NAMES.get(model_idx)
        if not model_name_lower:
            print(f"Error: Model index {model_idx} not defined in MODEL_NAMES. Skipping.")
            continue

        for snr_idx in SNR_INDICES:
            # 해당 SNR/모델 조합의 데이터를 모두 저장할 리스트
            all_case_data = []

            # 1. 10개의 케이스 (Seed) 데이터를 수집
            for case_num in CASE_RANGE:

                # 디렉토리 이름 생성 (예: rayleigh_itransformer_MSE)
                sub_dir = SUB_DIR_TEMPLATE.format(model_name_lower=model_name_lower)

                # 파일 경로 생성 (예: results/performance_test/case1.1.1/rayleigh_itransformer_MSE/performance_statistics.csv)
                file_path = BASE_PATH_TEMPLATE.format(
                    case_num=case_num,
                    model_idx=model_idx,
                    snr_idx=snr_idx,
                    sub_dir=sub_dir
                )

                if os.path.exists(file_path):
                    try:
                        df_case = pd.read_csv(file_path)
                        all_case_data.append(df_case)
                    except Exception as e:
                        print(f"경고: 파일 {file_path} 로드 실패: {e}")
                # else:
                #     print(f"경고: 파일 {file_path}를 찾을 수 없습니다.")

            # 2. 데이터가 있을 경우에만 평균 계산
            if all_case_data:
                # 10개 파일의 데이터를 하나의 DataFrame으로 결합
                combined_df = pd.concat(all_case_data, ignore_index=True)

                # Feature와 Metric을 기준으로 그룹화하고 'Mean' 컬럼의 평균을 계산
                # (각 파일의 'Mean' 컬럼을 평균내어 최종 평균을 구함)
                avg_results = combined_df.groupby(['Feature', 'Metric'])['Mean'].mean().reset_index()

                # 최종 결과를 딕셔너리에 저장
                key = f"Model_{model_idx}_SNRidx_{snr_idx}"
                final_averages[key] = avg_results
            else:
                print(f"경고: {model_idx}-{snr_idx} 조합에 대해 유효한 데이터를 찾지 못했습니다.")

    return final_averages

import matplotlib as mpl

def plot_snr_vs_mse_by_feature(csv_path, metric='MSE'):
    """
    단일 CSV 파일 경로를 받아 Feature별로 분리된 로그 스케일 MSE 그래프를 저장합니다.
    (CSV 파일은 이미 10개 Seed의 평균 결과가 포함되어 있다고 가정)

    Args:
        csv_path (str): 평균 성능 통계가 포함된 CSV 파일의 경로.
    """
    try:
        # 실제 코드에서는 df = pd.read_csv(csv_path)를 사용해야 합니다.
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 1. 데이터 전처리
    df['Feature'] = df['Feature'].str.replace('_measured', '').str.replace('_load', '').str.replace('_', ' ').str.title()
    df[['Model_ID', 'SNR_Index_Str']] = df['Case'].str.extract(r'Model_(\d)_SNRidx_(\d)')
    df['Model_ID'] = df['Model_ID'].astype(int)
    df['SNR_dB'] = (df['Model_ID'] - 1) * 3 # SNR Index (2~8) -> SNR dB (3~21)
    df['SNR_dB'] = (df['SNR_Index_Str'].astype(int) - 1) * 3

    # MSE 데이터 및 평균 계산
    df_mse = df[df['Metric'] == metric].copy()

    # 전체 피쳐 평균 MSE 계산
    df_overall_avg = df_mse.groupby(['Model_ID', 'SNR_dB'])['Mean'].mean().reset_index()
    df_overall_avg['Feature'] = 'Overall Average ' + metric

    # 2. 플로팅 디렉토리 생성
    output_dir = 'snr_mse_feature_plots'
    os.makedirs(output_dir, exist_ok=True)

    model_names = {1: 'Model 1 (DeepSC/IT)', 2: 'Model 2 (LSTM/GRU)'}

    # 3. 플로팅
    # 전체 Feature 목록 (Overall Average 포함)
    all_features = sorted(df_mse['Feature'].unique())
    features_to_plot = all_features + ['Overall Average ' + metric]

    # 플롯 스타일 정의
    styles = {
        1: {'marker': 'o', 'linestyle': '-'},
        2: {'marker': 's', 'linestyle': ':'}
    }

    # Matplotlib 컬러맵 초기화
    cmap = mpl.colormaps['tab10']

    for feature in features_to_plot:
        fig, ax = plt.subplots(figsize=(8, 5))

        # 해당 피쳐 데이터 필터링
        df_feature_plot = df_mse[df_mse['Feature'] == feature].copy()

        # 'Overall Average MSE'인 경우, df_overall_avg 데이터 사용
        if feature == 'Overall Average '+metric:
            df_feature_plot = df_overall_avg.copy()
            title = f'Overall Average {metric} Performance'
        else:
            # 개별 피쳐의 경우 모든 모델의 데이터를 모아야 함
            title = f'{metric} Performance vs. SNR for {feature}'

        for model_id in df_feature_plot['Model_ID'].unique():
            df_model = df_feature_plot[df_feature_plot['Model_ID'] == model_id]

            style = styles[model_id]

            if feature == f'Overall Average {metric}':
                # 전체 평균 스타일
                color = 'k' if model_id == 1 else 'gray'
                label = model_names[model_id]
                lw = 3
                ls = '--'
            else:
                # 개별 피쳐 스타일
                color = cmap(model_id)
                label = model_names[model_id]
                lw = 1.5
                ls = style['linestyle']

            ax.plot(df_model['SNR_dB'], df_model['Mean'],
                    label=label, color=color, linewidth=lw, linestyle=ls, marker=style['marker'])

        # 축 설정
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('SNR (dB)', fontsize=12)
        ax.set_ylabel(f'Mean Squared Error ({metric})', fontsize=12)
        # ax.set_yscale('log') # 로그 스케일 적용

        ax.grid(True, which="both", ls='--', alpha=0.6)
        ax.legend(loc='best', fontsize=10)
        ax.set_xticks(df_feature_plot['SNR_dB'].unique())

        # 파일명 저장
        safe_feature_name = feature.replace(' ', '_')
        plot_filename = os.path.join(output_dir, f'snr_{metric}_{safe_feature_name}.png')
        plt.savefig(plot_filename)
        plt.close(fig)
        print(f"Saved plot: {plot_filename}")

    return f"총 {len(features_to_plot)}개의 피쳐별 {metric} 그래프가 '{output_dir}' 디렉토리에 저장되었습니다."

# 실행
if __name__ == "__main__":
    avg_performance = calculate_average_performance()

    # 최종 결과 출력
    for key, df_result in avg_performance.items():
        print(f"\n==============================================")
        print(f"평균 결과: {key}")
        print("==============================================")
        print(df_result.to_markdown(index=False))

    # (선택 사항) 결과를 CSV 파일로 저장
    avg_results_df = pd.concat([df.assign(Case=key) for key, df in avg_performance.items()], ignore_index=True)
    avg_results_df.to_csv("average_performance_summary.csv", index=False)
    print("\n전체 평균 결과가 average_performance_summary.csv에 저장되었습니다.")

    # SNR vs MSE 플롯 생성
    plot_file = plot_snr_vs_mse_by_feature("average_performance_summary.csv", metric = "MAE")
    print(f"플롯이 '{plot_file}' 파일로 저장되었습니다.")
