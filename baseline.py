import numpy as np
import gzip
import pickle
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

from parameters.parameters import outlier_cut_csv_path

import pdb

# -----------------------------
#  Utility: Noise Injection
# -----------------------------
def add_noise_to_bytes(data_bytes: bytes, noise_std: float) -> bytes:
    """
    Add Gaussian noise to compressed representation (bytes → float array → noisy → back to bytes).
    """
    arr = np.frombuffer(data_bytes, dtype=np.uint8).astype(np.float32)
    noise = np.random.normal(0, noise_std, size=arr.shape)
    noisy_arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return noisy_arr.tobytes()

# -----------------------------
#  Gorilla-like (delta encoding + XOR)
# -----------------------------
def gorilla_compress(data: np.ndarray, noise_std: float = 0.0):
    data_int = (data * 1e6).astype(np.int64)  # scale floats to int
    deltas = np.diff(data_int, axis=0, prepend=data_int[0:1])
    compressed = pickle.dumps(deltas)

    if noise_std > 0:
        compressed = add_noise_to_bytes(compressed, noise_std)

    return compressed

def gorilla_decompress(comp_bytes: bytes, shape):
    deltas = pickle.loads(comp_bytes)
    recovered = np.cumsum(deltas, axis=0)
    return recovered.astype(np.float64) / 1e6

# -----------------------------
#  ARIMA + residual coding
# -----------------------------
def arima_compress(data: np.ndarray, noise_std: float = 0.0, order=(1,0,0)):
    residuals = []
    models = []
    for i in range(data.shape[1]):
        series = data[:, i]
        model = ARIMA(series, order=order)
        fit = model.fit()
        pred = fit.fittedvalues
        res = series - pred
        residuals.append(res)
        models.append(fit)

    compressed = pickle.dumps((models, residuals))

    if noise_std > 0:
        compressed = add_noise_to_bytes(compressed, noise_std)

    return compressed

def arima_decompress(comp_bytes: bytes):
    models, residuals = pickle.loads(comp_bytes)
    recovered = []
    for fit, res in zip(models, residuals):
        pred = fit.fittedvalues
        series = pred + res
        recovered.append(series)
    return np.vstack(recovered).T

# -----------------------------
#  Gzip (lossless generic)
# -----------------------------
def gzip_compress(data: np.ndarray, noise_std: float = 0.0):
    raw_bytes = pickle.dumps(data)
    comp = gzip.compress(raw_bytes)

    if noise_std > 0:
        comp = add_noise_to_bytes(comp, noise_std)

    return comp

def gzip_decompress(comp_bytes: bytes):
    raw_bytes = gzip.decompress(comp_bytes)
    return pickle.loads(raw_bytes)

# -----------------------------
#  Dispatcher
# -----------------------------
def compress(data, method="gorilla", noise_std=0.0):
    if method == "gorilla":
        return gorilla_compress(data, noise_std)
    elif method == "arima":
        return arima_compress(data, noise_std)
    elif method == "gzip":
        return gzip_compress(data, noise_std)
    else:
        raise ValueError("Unknown method")

def decompress(comp_bytes, method, shape=None):
    if method == "gorilla":
        return gorilla_decompress(comp_bytes, shape)
    elif method == "arima":
        return arima_decompress(comp_bytes)
    elif method == "gzip":
        return gzip_decompress(comp_bytes)
    else:
        raise ValueError("Unknown method")

# -----------------------------
#  Run Experiment and Save Results
# -----------------------------
def run_experiment(data: np.ndarray, methods=["gorilla", "arima", "gzip"], noise_std=0.0, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)

    metrics = []
    for m in methods:
        print(f"\nMethod: {m}")
        if m == "gorilla" : continue
        comp = compress(data, method=m, noise_std=noise_std)
        rec = decompress(comp, method=m, shape=data.shape)
        pdb.set_trace()

        mse = mean_squared_error(data, rec)
        mae = mean_absolute_error(data, rec)
        metrics.append({"method": m, "MSE": mse, "MAE": mae})

        # Save recovered data to CSV
        rec_df = pd.DataFrame(rec, columns=[f"feature_{i}" for i in range(data.shape[1])])
        rec_path = os.path.join(save_dir, f"recovered_{m}.csv")
        rec_df.to_csv(rec_path, index=False)

        print("Recovered shape:", rec.shape)
        print(f"MSE: {mse:.6f}, MAE: {mae:.6f}")

    # Save metrics summary
    metrics_df = pd.DataFrame(metrics)
    metrics_path = os.path.join(save_dir, "metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print("\nMetrics saved to:", metrics_path)

if __name__ == "__main__":
    df = pd.read_csv(os.path.join(outlier_cut_csv_path,"01295.csv"))
    run_experiment(df, methods=["gorilla", "arima", "gzip"], noise_std=2.0, save_dir="results")
