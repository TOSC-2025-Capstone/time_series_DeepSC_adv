import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

MERGED_DIR = "data_handling/merged"
WINDOW_SIZES = [32, 64, 128, 256]
CYCLE_COL = 'cycle_idx'
FEATURE_COLS = [
    'Voltage_measured', 'Current_measured', 'Temperature_measured', 'Current_load', 'Voltage_load', 'Time'
]

SAVE_DIR = "./analysis/cycle_window_stats/"
os.makedirs(SAVE_DIR, exist_ok=True)

def get_cycle_length_stats():
    cycle_lengths = []
    for fname in os.listdir(MERGED_DIR):
        if not fname.endswith('.csv'):
            continue
        fpath = os.path.join(MERGED_DIR, fname)
        try:
            df = pd.read_csv(fpath)
            if CYCLE_COL not in df.columns:
                continue
            for cycle_id, group in df.groupby(CYCLE_COL):
                cycle_lengths.append(len(group))
        except Exception as e:
            print(f"[WARN] {fname}: {e}")
    cycle_lengths = np.array(cycle_lengths)
    print("=== Cycle Length Statistics ===")
    print(f"Total cycles: {len(cycle_lengths)}")
    print(f"Mean: {cycle_lengths.mean():.2f}")
    print(f"Median: {np.median(cycle_lengths):.2f}")
    print(f"Min: {cycle_lengths.min()}  Max: {cycle_lengths.max()}")
    plt.figure(figsize=(8,5))
    plt.hist(cycle_lengths, bins=30, edgecolor='black', alpha=0.7)
    plt.title('Cycle Length Distribution')
    plt.xlabel('Cycle Length (row count)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'cycle_length_hist.png'))
    plt.show()
    return cycle_lengths

def get_window_count_stats(window_sizes):
    window_counts = {ws: [] for ws in window_sizes}
    for fname in os.listdir(MERGED_DIR):
        if not fname.endswith('.csv'):
            continue
        fpath = os.path.join(MERGED_DIR, fname)
        try:
            df = pd.read_csv(fpath)
            if CYCLE_COL not in df.columns:
                continue
            for cycle_id, group in df.groupby(CYCLE_COL):
                cycle_len = len(group)
                for ws in window_sizes:
                    n_win = max(0, cycle_len - ws + 1)
                    window_counts[ws].append(n_win)
        except Exception as e:
            print(f"[WARN] {fname}: {e}")
    # 통계 및 시각화
    for ws in window_sizes:
        arr = np.array(window_counts[ws])
        print(f"\n=== Window Size {ws} ===")
        print(f"Total cycles: {len(arr)}")
        print(f"Cycles with >=1 window: {(arr>0).sum()} ({(arr>0).sum()/len(arr)*100:.1f}%)")
        print(f"Mean windows per cycle: {arr[arr>0].mean() if (arr>0).sum()>0 else 0:.2f}")
        plt.figure(figsize=(8,5))
        plt.hist(arr[arr>0], bins=30, edgecolor='black', alpha=0.7)
        plt.title(f'Window Count per Cycle (window_size={ws})')
        plt.xlabel('Number of windows')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, f'window_count_hist_ws{ws}.png'))
        plt.show()
    # 전체 bar plot
    total_windows = [np.sum(window_counts[ws]) for ws in window_sizes]
    plt.figure(figsize=(7,5))
    plt.bar([str(ws) for ws in window_sizes], total_windows, color='skyblue')
    plt.title('Total Window Count by Window Size')
    plt.xlabel('Window Size')
    plt.ylabel('Total Number of Windows')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'total_window_count_bar.png'))
    plt.show()
    return window_counts

def main():
    print("\n[Cycle Length Stats]")
    cycle_lengths = get_cycle_length_stats()
    print("\n[Window Count Stats]")
    window_counts = get_window_count_stats(WINDOW_SIZES)

if __name__ == "__main__":
    main() 