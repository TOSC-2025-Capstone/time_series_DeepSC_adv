import pdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from tqdm import tqdm
import re
import os

# --- 폰트 설정 (영어 및 기본 설정) ---
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
flag = 0

class CycleDataComparator:
    def __init__(self, original_path, reconstructed_path, threshold_percent=15, threshold_method="mean", min_absolute_threshold=1e-9):
        if threshold_method not in ["mean", "range_center", "point_wise"]:
            raise ValueError("threshold_method must be one of 'mean', 'range_center', or 'point_wise'")

        self.original_path = Path(original_path)
        self.reconstructed_path = Path(reconstructed_path)
        self.threshold_percent = threshold_percent
        self.threshold_method = threshold_method
        self.min_absolute_threshold = min_absolute_threshold
        self.threshold_ratio = self.threshold_percent / 100.0

        self.target_features = ['Voltage_measured', 'Current_measured', 'Temperature_measured', 'Current_load', 'Voltage_load']
        self.voltage_features = ['Voltage_measured', 'Voltage_load']
        self.current_features = ['Current_measured', 'Current_load']

        self.discrete_configs = {
            'Current_measured': {'levels': np.array([-4, -2, -1, 0]), 'midpoints': np.array([-3.0, -1.5, -0.5])},
            'Current_load': {'levels': np.array([-2, 0, 1, 2, 4]), 'midpoints': np.array([-1.0, 0.5, 1.5, 3.0])}
        }

    def get_reconstructed_files(self):
        return sorted(self.reconstructed_path.glob('*_reconstructed.csv'))

    def load_data_pair(self, reconstructed_file):
        file_name = reconstructed_file.stem
        file_number = file_name.split('_reconstructed')[0]
        original_file = self.original_path / f"{file_number}.csv"
        if not original_file.exists(): return None, None, file_number
        try:
            return pd.read_csv(original_file), pd.read_csv(reconstructed_file), file_number
        except Exception: return None, None, file_number

    def perform_hard_decision(self, recon_series, config):
        indices = np.digitize(recon_series, config['midpoints'])
        return config['levels'][indices]

    def compare_data(self):
        reconstructed_files = self.get_reconstructed_files()
        total_mse_list, total_rows_global = [], 0
        feature_fail_counts = {f: 0 for f in self.target_features}

        if not reconstructed_files:
            return 0, 0, 0, 0, feature_fail_counts

        for reconstructed_file in tqdm(reconstructed_files, desc="Comparing", leave=False):
            orig_df, recon_df, _ = self.load_data_pair(reconstructed_file)
            if orig_df is None: continue

            n_rows = min(len(orig_df), len(recon_df))
            orig_df, recon_df = orig_df.iloc[:n_rows].reset_index(drop=True), recon_df.iloc[:n_rows].reset_index(drop=True)

            file_mse_sum = 0
            for feature in self.target_features:
                if feature not in orig_df.columns: continue
                diff = (orig_df[feature] - recon_df[feature]).abs()
                file_mse_sum += (diff**2).mean()

                if feature in self.discrete_configs:
                    config = self.discrete_configs[feature]
                    success_mask = (self.perform_hard_decision(orig_df[feature], config) == self.perform_hard_decision(recon_df[feature], config))
                    # global flag
                    # if flag == 1 and success_mask.sum() <= len(orig_df[feature])-10 :
                    #     pdb.set_trace()
                elif feature in self.voltage_features:
                    success_mask = diff <= (orig_df[feature].max() * self.threshold_ratio)
                else:
                    ref = abs(orig_df[feature].mean()) if self.threshold_method == "mean" else (orig_df[feature].max() - orig_df[feature].min())/2
                    success_mask = diff <= (ref * self.threshold_ratio)
                feature_fail_counts[feature] += (~success_mask).sum()

            total_mse_list.append(file_mse_sum / len(self.target_features))
            total_rows_global += n_rows

        total_failed_cells = sum(feature_fail_counts.values())
        total_cells = total_rows_global * len(self.target_features)
        cell_fail_rate = (total_failed_cells / total_cells * 100) if total_cells > 0 else 0
        return np.mean(total_mse_list), cell_fail_rate, total_failed_cells, total_rows_global, feature_fail_counts

    def analyze_reconstructed_distributions(self, model_name, snr_val):
        """모든 테스트 파일의 Current 복원값 분포 수집 및 시각화"""
        reconstructed_files = self.get_reconstructed_files()
        collected_values = {feat: [] for feat in self.current_features}

        for recon_file in tqdm(reconstructed_files, desc=f"Collecting {model_name}", leave=False):
            _, recon_df, _ = self.load_data_pair(recon_file)
            if recon_df is None: continue
            for feat in self.current_features:
                if feat in recon_df.columns:
                    collected_values[feat].extend(recon_df[feat].tolist())

        for feat in self.current_features:
            data = collected_values[feat]
            if not data: continue
            plt.figure(figsize=(10, 6))
            plt.hist(data, bins=100, color='darkgray', edgecolor='white', alpha=0.8, density=True)
            if feat in self.discrete_configs:
                for level in self.discrete_configs[feat]['levels']:
                    plt.axvline(x=level, color='red', linestyle='--', linewidth=1, alpha=0.6)
            plt.title(f'Distribution of Reconstructed {feat} ({model_name}, SNR {snr_val}dB)', fontsize=16, pad=15)
            plt.xlabel('Reconstructed Value', fontsize=14)
            plt.ylabel('Density', fontsize=14)
            plt.grid(axis='y', linestyle=':', alpha=0.5)
            output_dir = Path(f"./comparison_results/distributions/{model_name}")
            output_dir.mkdir(parents=True, exist_ok=True)
            save_path = output_dir / f"dist_{feat}_snr{snr_val}.png"
            plt.tight_layout()
            plt.savefig(save_path, dpi=300)
            plt.close()

    def save_results(self, output_base, avg_mse, fail_rate, fail_count, total_rows, feature_fails):
        output_base = Path(output_base)
        threshold_folder = output_base / f"threshold_{self.threshold_percent}percent_{self.threshold_method}"
        threshold_folder.mkdir(parents=True, exist_ok=True)

        txt_path = threshold_folder / "summary_feature_wise_text.txt"
        total_cells = total_rows * len(self.target_features)
        # 텍스트 리포트에도 정확도 정보 추가
        accuracy = 100 - fail_rate

        lines = ["="*80, f"SUMMARY REPORT - Threshold: {self.threshold_percent}% ({self.threshold_method})", f"Logic: Cell-wise Accuracy (Total Cells: {total_cells})", "="*80, "\nFailure Counts by Feature (Cell-wise):", "-"*80]
        for f in self.target_features:
            count = feature_fails.get(f, 0)
            percent = (count / total_rows * 100) if total_rows > 0 else 0
            lines.append(f"{f:30s}: {count:8d} / {total_rows:8d} ({percent:6.2f}%)")
        lines.append("\nOverall Cell-wise Results:")
        lines.append("-"*80)
        lines.append(f"{'Failed Cells':30s}: {fail_count:8d} / {total_cells:8d} ({fail_rate:6.2f}%)")
        lines.append(f"{'Accuracy':30s}: {total_cells - fail_count:8d} / {total_cells:8d} ({accuracy:6.2f}%)")
        lines.append("-"*80)
        lines.append(f"Average MSE Across Features: {avg_mse:.6f}\n" + "="*80)

        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
        print(f"   [SUCCESS] Text summary saved to: {txt_path}")
        return threshold_folder

def plot_individual_results(df, output_dir, snr_list):
    """범례 순서, 색상 진도, 마커 스타일이 최적화된 그래프 생성"""

    # [1] 범례 표시 순서 설정 (사용자 요청: LSTM -> GRU -> Transformer -> Inverted-Transformer)
    model_names_ordered = ['LSTM', 'GRU', 'Transformer', 'Inverted-Transformer']

    # [2] Grayscale 테마 설정
    # np.linspace(0.3, 0.9)를 사용하여 뒤로 갈수록(Inverted-Transformer 쪽) 색이 진해짐 (0.9가 가장 진함)
    cmap = cm.get_cmap('Greys')
    grayscale_colors = [cmap(i) for i in np.linspace(0.3, 0.9, len(model_names_ordered))]

    # [3] 마커 설정 (사용자 요청: Inverted-Transformer가 동그라미 'o'가 되도록 마지막에 배치)
    # 순서대로 LSTM: 's', GRU: '^', Transformer: 'D', Inverted-Transformer: 'o'
    markers = ['s', '^', 'D', 'o']
    linestyles = ['--', ':', '-.', '-'] # Inverted-Transformer를 실선(-)으로 강조

    # 모델별 스타일 매핑
    style_map = {model: {'color': grayscale_colors[i],
                         'marker': markers[i],
                         'linestyle': linestyles[i]}
                 for i, model in enumerate(model_names_ordered)}

    metrics = {
        'MSE': ('Reconstruction MSE vs. SNR', 'Average MSE', 'model_comparison_MSE.png'),
        'Fail_Rate': ('Cell-wise Accuracy vs. SNR', 'Cell-wise Accuracy (%)', 'model_comparison_Accuracy.png')
    }

    for column, (title, ylabel, filename) in metrics.items():
        fig, ax = plt.subplots(figsize=(10, 6))

        for model in model_names_ordered:
            subset = df[df['Model'] == model].sort_values(by='SNR')
            if subset.empty: continue
            style = style_map[model]

            # 정확도 변환 로직 (필요 시)
            plot_data = subset[column]
            if column == 'Fail_Rate':
                plot_data = 100 - plot_data

            ax.plot(subset['SNR'], plot_data,
                    color=style['color'],
                    marker=style['marker'],
                    linestyle=style['linestyle'],
                    label=model,
                    linewidth=2.5,  # 가시성을 위해 선 두께 소폭 강화
                    markersize=9)

        # 타이틀 및 레이블 설정 (볼드체 제거 유지)
        ax.set_title(title, fontsize=16, pad=15)
        ax.set_xlabel('SNR (dB)', fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_xticks(snr_list)
        ax.tick_params(axis='both', which='major', labelsize=11)
        ax.grid(True, linestyle='--', alpha=0.6)

        # 범례 설정: 상단 요청 순서대로 표시됨
        ax.legend(title='Model', fontsize=12, title_fontsize=13, loc='best', frameon=True, shadow=True)

        plt.tight_layout()
        save_path = Path(output_dir) / filename
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"   [GRAPH] Saved: {save_path}")

def main():
    global flag
    case_id_prefix = "10045"
    orig_path = "./cycle_preprocess/csv/outlier_cut/threshold_7/cycle_len_512"
    model_map = {'1': 'Inverted-Transformer', '2': 'LSTM', '3': 'GRU', '4': 'Transformer'}
    snr_map = {'2': 3, '3': 6, '4': 9, '5': 12, '6': 15, '7': 18, '8': 21}
    model_type_map = {'1': 'deepsc', '4': 'deepsc', '2': 'lstm', '3': 'gru'}

    aggregated_results = []
    snr_values = sorted(list(snr_map.values()))

    for m_idx, m_name in model_map.items():
        for s_idx, s_val in snr_map.items():
            model_type = model_type_map.get(m_idx, "unknown")
            recon_path = f"./reconstruction/case{case_id_prefix}.{m_idx}.{s_idx}/reconstructed_AWGN_{model_type}_MSE"
            if not Path(recon_path).exists(): continue

            print(f"\nAnalyzing: {m_name} | SNR: {s_val}dB")
            comparator = CycleDataComparator(orig_path, recon_path, threshold_percent=10, threshold_method="mean")
            if s_idx == '8':
                flag = 1
            avg_mse, cell_fail_rate, total_f_cells, t_rows, f_fails = comparator.compare_data()

            flag = 0

            output_dir = f"./comparison_results/case_{case_id_prefix}_{m_name}_snr{s_val}"
            comparator.save_results(output_dir, avg_mse, cell_fail_rate, total_f_cells, t_rows, f_fails)

            # 전류 분포 분석 (필요 시 주석 처리)
            # if s_idx == '8':  # SNR 21dB에서만 분석
            #     comparator.analyze_reconstructed_distributions(m_name, s_val)

            aggregated_results.append({'Model': m_name, 'SNR': s_val, 'MSE': avg_mse, 'Fail_Rate': cell_fail_rate})

    if aggregated_results:
        results_df = pd.DataFrame(aggregated_results)
        # 개별 파일 저장 함수 호출
        plot_individual_results(results_df, "./comparison_results", snr_values)

if __name__ == "__main__":
    main()
