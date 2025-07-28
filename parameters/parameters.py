from dataclasses import dataclass, field
from models.lstm import LSTMDeepSC
from models.gru import GRUDeepSC
from models.attention_lstm import LSTMAttentionDeepSC
from models.transceiver import DeepSC

import torch
from typing import List

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 처음 실행 여부 -> True면 이미 전처리된 데이터로 학습 및 평가 진행, False면 preprocess 실행
is_preprocessed = True
# is_preprocessed = False

# 학습이 완료되었는지 여부 -> True면 학습된 모델로 평가, False면 학습 진행
# is_trained = True
is_trained = False

# window arr
window_arr = [32, 64, 128, 256]
# models arr
models_type_arr = ["deepsc", "lstm", "gru", "at_lstm"]
case_index = 7.3
loss_type = "MSE"
model_select = 2
model_type = models_type_arr[model_select]
channel_type = "no_channel"

# scaler_type
scaler_type = "minmax"  # or "zscore"

# feature cols (inputs)
feature_cols = [
    "Voltage_measured",
    "Current_measured",
    "Temperature_measured",
    "Current_load",
    "Voltage_load",
    "Time",
]

# 이상치 제거 여부 -> 아래 경로 지정에 쓰임
is_outlier_cut = False

# 전처리 입력으로 사용할 데이터 경로 (merged)
original_data_path = "./original_data/cycle_data/"
# 중간에 이상치 제거 버전 저장할 경로 -> 나중에 이걸 csv 복원 비교의 원본 csv으로 사용
outlier_cut_csv_path = (
    # f"./data/case_{case_index}/merged{'_outlier_cut' if is_outlier_cut else ''}"
    "./cycle_preprocess/csv/outlier_cut/"
)
# merged의 파일에서 이상치가 제거되며 전처리 된 데이터 경로 (train_data.pt, test_data.pt)
preprocessed_data_path = f"./cycle_preprocess/total_preprocessed/processed_minmax"
# 모델 저장 경로
model_checkpoint_path = f"./checkpoints/case_{case_index}/{loss_type}/{model_type}/{model_type}_battery_epoch"
# 복원 후 데이터 경로
reconstructed_data_path = f"./reconstruction/case_{case_index}/reconstructed_{channel_type}_{model_type}_{loss_type}"

## model params
# epochs
train_epochs = 80
# batch size
train_batch_size = 32
# learning rate
tratin_lr = 1e-5
# input dimension
input_dim = 6
# window size
window_size = window_arr[3]
# stride
stride = window_size // 2

## reconstruction save path
save_fig_dir = f"results/case{case_index}/{channel_type}_{model_type}_{loss_type}"
save_reconstruct_dir = f"reconstruction/case{case_index}/reconstructed_{channel_type}_{model_type}_{loss_type}"
save_performance_dir = (
    f"results/performance_test/case{case_index}/{channel_type}_{model_type}_{loss_type}"
)


# preprocess
@dataclass(unsafe_hash=True)
class PreprocessParams:
    folder_path: str = original_data_path
    feature_cols: List[str] = field(default_factory=lambda: feature_cols.copy())
    batch_size: int = 8
    save_split_path: str = preprocessed_data_path
    split_ratio: float = 0.8
    window_size: int = window_size
    stride: int = 32
    sample_num: int = 1000
    PREPROCESSED_DIR: str = outlier_cut_csv_path


# train
@dataclass
class TrainDeepSCParams:
    train_pt: str = preprocessed_data_path + "/train_data.pt"
    validate_pt: str = preprocessed_data_path + "/val_data.pt"
    scaler_path: str = preprocessed_data_path + "/scaler.pkl"
    model_save_path: str = model_checkpoint_path
    num_epochs: int = train_epochs
    batch_size: int = train_batch_size
    lr: float = tratin_lr


# test
@dataclass
class TestParams:
    loss_type: str = loss_type
    model_type: str = model_type
    channel_type: str = channel_type
    csv_origin_path: str = outlier_cut_csv_path  # 복원본과 비교할 대상 (이상치제거본)
    preprocessed_path: str = preprocessed_data_path
    save_performance_dir = save_performance_dir
    save_reconstruct_dir = save_reconstruct_dir
    feature_cols: List[str] = field(default_factory=lambda: feature_cols.copy())
    train_pt = preprocessed_data_path + "/train_data.pt"
    test_pt = preprocessed_data_path + "/test_data.pt"
    scaler_path = preprocessed_data_path + "/scaler.pkl"
