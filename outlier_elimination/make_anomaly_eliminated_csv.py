import os
import pandas as pd
import numpy as np

SRC_DIR = "data/merged"
DST_DIR = "data/merged_anomaly_eliminated_z"
FEATURE_COLS = [
    'Voltage_measured', 'Current_measured', 'Temperature_measured', 'Current_load', 'Voltage_load', 'Time'
]

os.makedirs(DST_DIR, exist_ok=True)

def correct_outliers_with_interpolation_zscore(df, feature_cols, z_thresh=3):
    df = df.copy()
    for col in feature_cols:
        series = df[col]
        zscore = (series - series.mean()) / series.std(ddof=0)
        outlier_mask = np.abs(zscore) >= z_thresh
        if outlier_mask.any():
            series[outlier_mask] = np.nan
            series = series.interpolate(method='linear', limit_direction='both')
            series = series.fillna(method='bfill').fillna(method='ffill')
            df[col] = series
    return df

def main():
    files = [f for f in os.listdir(SRC_DIR) if f.endswith('.csv')]
    for fname in files:
        src_path = os.path.join(SRC_DIR, fname)
        dst_path = os.path.join(DST_DIR, fname)
        try:
            df = pd.read_csv(src_path)
            df_corr = correct_outliers_with_interpolation_zscore(df, FEATURE_COLS, z_thresh=7)
            df_corr.to_csv(dst_path, index=False)
            print(f"[OK] {fname} → {dst_path}")
        except Exception as e:
            print(f"[FAIL] {fname}: {e}")

if __name__ == "__main__":
    main() 