import pdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as mticker
from pathlib import Path
from tqdm import tqdm
import textwrap
import os

# --- 폰트 및 스타일 설정 ---
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

class CycleDataComparator:
    def __init__(self, original_path, reconstructed_path, threshold_percent=1, threshold_method="mean"):
        self.original_path = Path(original_path)
        self.reconstructed_path = Path(reconstructed_path)
        self.threshold_ratio = threshold_percent / 100.0
        self.threshold_method = threshold_method
        self.mape_threshold = 0.01  # 사용자가 요청한 MAPE 계산 제외 임계값

        self.target_features = ['Voltage_measured', 'Current_measured', 'Temperature_measured', 'Current_load', 'Voltage_load']
        self.discrete_configs = {
            'Current_measured': {'levels': np.array([-4, -2, -1, 0]), 'midpoints': np.array([-3.0, -1.5, -0.5])},
            'Current_load': {'levels': np.array([-2, 0, 1, 2, 4]), 'midpoints': np.array([-1.0, 0.5, 1.5, 3.0])}
        }

    def get_reconstructed_files(self):
        return sorted(self.reconstructed_path.glob('*_reconstructed.csv'))

    def load_data_pair(self, reconstructed_file):
        file_number = reconstructed_file.stem.split('_reconstructed')[0]
        original_file = self.original_path / f"{file_number}.csv"
        if not original_file.exists(): return None, None
        return pd.read_csv(original_file), pd.read_csv(reconstructed_file)

    def perform_hard_decision(self, recon_series, config):
        indices = np.digitize(recon_series, config['midpoints'])
        return config['levels'][indices]

    def compare_data(self):
        reconstructed_files = self.get_reconstructed_files()
        total_mse_list = []
        total_mape_list = []
        total_rows_global = 0
        feature_fail_counts = {f: 0 for f in self.target_features}

        for recon_file in tqdm(reconstructed_files, desc="Processing Files", leave=False):
            orig_df, recon_df = self.load_data_pair(recon_file)
            if orig_df is None: continue

            n_rows = min(len(orig_df), len(recon_df))
            orig_df, recon_df = orig_df.iloc[:n_rows], recon_df.iloc[:n_rows]

            file_mse = []
            file_mape = []

            for feature in self.target_features:
                if feature not in orig_df.columns: continue
                y_true = orig_df[feature].values
                y_pred = recon_df[feature].values
                diff = np.abs(y_true - y_pred)

                # 1. MSE 계산
                file_mse.append(np.mean(diff**2))

                # 2. 정확도(Accuracy) 판정 로직
                if feature in self.discrete_configs:
                    config = self.discrete_configs[feature]
                    success_mask = (self.perform_hard_decision(y_true, config) == self.perform_hard_decision(y_pred, config))
                elif 'Voltage' in feature:
                    success_mask = diff <= (np.max(y_true) * self.threshold_ratio)
                else:
                    ref = np.mean(np.abs(y_true)) if self.threshold_method == "mean" else (np.max(y_true)-np.min(y_true))/2
                    success_mask = diff <= (ref * self.threshold_ratio)

                feature_fail_counts[feature] += np.sum(~success_mask)

                # 3. MAPE 계산 (임계값 0.1 기준 필터링)
                mape_mask = np.abs(y_true) >= self.mape_threshold
                if np.any(mape_mask):
                    mape_val = np.mean(np.abs((y_true[mape_mask] - y_pred[mape_mask]) / y_true[mape_mask])) * 100
                    file_mape.append(mape_val)

            total_mse_list.append(np.mean(file_mse))
            if file_mape: total_mape_list.append(np.mean(file_mape))
            total_rows_global += n_rows

        total_cells = total_rows_global * len(self.target_features)
        accuracy = (1 - (sum(feature_fail_counts.values()) / total_cells)) * 100
        avg_mape = np.mean(total_mape_list) if total_mape_list else 0

        return np.mean(total_mse_list), accuracy, avg_mape

def plot_combined_metrics(df, output_dir, snr_list):
    """정확도(왼쪽 Y축)와 MAPE(오른쪽 Y축)를 4행 2열 범례로 시각화"""
    model_order = ['LSTM', 'GRU', 'Transformer', 'Inverted-Transformer']
    markers = ['s', '^', 'D', 'o']
    linestyles = ['--', ':', '-.', '-']
    cmap = cm.get_cmap('Greys')
    colors = [cmap(i) for i in np.linspace(0.4, 0.9, len(model_order))]

    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax2 = ax1.twinx()

    acc_handles = []
    mape_handles = []

    for i, model in enumerate(model_order):
        subset = df[df['Model'] == model].sort_values(by='SNR')
        if subset.empty: continue

        # 정확도 (왼쪽 축)
        l1 = ax1.plot(subset['SNR'], subset['Accuracy'], color=colors[i], marker=markers[i],
                      linestyle=linestyles[i], linewidth=2.5, markersize=8, label=f"{model} (Acc)")
        acc_handles.append(l1[0])

        # MAPE (오른쪽 축)
        l2 = ax2.plot(subset['SNR'], subset['MAPE'], color=colors[i], marker=markers[i],
                      linestyle=(0, (1, 1)), linewidth=1.5, alpha=0.6, label=f"{model} (MAPE)")
        mape_handles.append(l2[0])

    # 범례 리스트 통합 (왼쪽 열: Acc 리스트, 오른쪽 열: MAPE 리스트)
    all_handles = acc_handles + mape_handles

    # 축 설정
    ax1.set_xlabel('SNR (dB)', fontsize=14)
    ax1.set_ylabel('Cell-wise Accuracy (%)', fontsize=14, color='black')
    ax2.set_ylabel('Global MAPE (%)', fontsize=14, color='dimgray')

    ax1.set_xticks(snr_list)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Y축 범위 최적화
    ax1.set_ylim(min(df['Accuracy'])*0.95, 101)
    ax2.set_ylim(0, max(df['MAPE'])*1.2)

    plt.title('Reconstruction Reliability: Accuracy vs. MAPE across SNR', fontsize=16, pad=20)

    # [수정 포인트] 범례를 2열(ncol=2)로 설정하고 열 간격(columnspacing) 조정
    ax1.legend(handles=all_handles,
               loc='lower right',
               ncol=2,
               fontsize=10,
               columnspacing=1.0,
               handletextpad=0.5,
               title='Models (Left: Acc | Right: MAPE)',
               title_fontsize=11,
               frameon=True,
               shadow=True)

    plt.tight_layout()
    save_path = Path(output_dir) / "combined_accuracy_mape_snr.png"
    plt.savefig(save_path, dpi=300)
    print(f"[GRAPH SAVED] {save_path}")

def main():
    orig_path = "./cycle_preprocess/csv/outlier_cut/threshold_7/cycle_len_512"
    model_map = {'2': 'LSTM', '3': 'GRU', '4': 'Transformer', '1': 'Inverted-Transformer'}
    # snr_map = {'2': 3, '3': 6, '4': 9, '5': 12, '6': 15, '7': 18, '8': 21}
    snr_map = {'1': 21}
    model_type_map = {'1': 'deepsc', '4': 'deepsc', '2': 'lstm', '3': 'gru'}

    aggregated_results = []
    snr_values = sorted(list(snr_map.values()))

    for m_idx, m_name in model_map.items():
        for s_idx, s_val in snr_map.items():
            model_type = model_type_map.get(m_idx, "unknown")
            # 경로 구조: case10052.model_idx.snr_idx
            recon_path = f"./reconstruction/case10055.{m_idx}.{s_idx}/reconstructed_AWGN_{model_type}_MSE"

            if not Path(recon_path).exists():
                print(f"Skipping: {recon_path} (Not found)")
                continue

            print(f"Analyzing {m_name} at SNR {s_val}dB...")
            comparator = CycleDataComparator(orig_path, recon_path, threshold_percent=3)
            avg_mse, accuracy, avg_mape = comparator.compare_data()

            aggregated_results.append({
                'Model': m_name, 'SNR': s_val, 'MSE': avg_mse,
                'Accuracy': accuracy, 'MAPE': avg_mape
            })

    if aggregated_results:
        results_df = pd.DataFrame(aggregated_results)
        output_dir = "./comparison_results"
        os.makedirs(output_dir, exist_ok=True)
        plot_combined_metrics(results_df, output_dir, snr_values)
        results_df.to_csv(f"{output_dir}/global_performance_summary.csv", index=False)

if __name__ == "__main__":
    main()
