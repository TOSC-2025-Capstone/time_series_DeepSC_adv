from dataclasses import dataclass, field

from models.lstm import LSTMDeepSC
from models.gru import GRUDeepSC  
from models.attention_lstm import LSTMAttentionDeepSC
from models.transceiver import DeepSC

import torch
from typing import List

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 처음 실행 여부 -> 처음 1번만 preprocess하고 나서는 실행 x
is_first = False

# window arr
window_arr = [32, 64, 128, 256] 
# models arr 
models_type_arr = ['deepsc', 'lstm', 'gru', 'at_lstm']
case_index = 4
loss_type = 'MSE'
model_select = 0
model_type = models_type_arr[model_select]
channel_type = 'no_channel'

# 전처리 입력으로 사용할 데이터 경로 (merged)
original_data_path = "./data/merged"
# 중간에 이상치 제거 버전 저장할 경로 -> 나중에 이걸 csv 복원 비교의 원본 csv으로 사용
outlier_cut_csv_path = "./data/merged_outlier_cut"
# merged의 파일에서 이상치가 제거되며 전처리 된 데이터 경로 (train_data.pt, test_data.pt)
preprocessed_data_path = "./preprocessed/preprocessed_data_0717_outlier_cut"
# 모델 저장 경로 
model_checkpoint_path = f"./checkpoints/case{case_index}/{loss_type}/{model_type}/{model_type}_battery_epoch"
# 복원 후 데이터 경로
reconstructed_data_path = f'./recons/case{case_index}/reconstructed_{channel_type}_{model_type}_{loss_type}'

# feature cols (inputs)
feature_cols = [
    'Voltage_measured', 'Current_measured', 'Temperature_measured', 'Current_load', 'Voltage_load', 'Time'
]

## model params
# epochs
train_epochs = 30
# batch size
train_batch_size = 32
# learning rate
tratin_lr = 1e-5
# input dimension
input_dim = 6
# window size
window_size = window_arr[2]
# stride
stride = window_size//4

## reconstruction save path
save_recon_dir = f'reconstruction/case{case_index}/reconstructed_{channel_type}_{model_type}_{loss_type}'
save_fig_dir = f'results/case{case_index}/{channel_type}_{model_type}_{loss_type}'

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
    train_pt: str = preprocessed_data_path+'/train_data.pt'
    test_pt: str = preprocessed_data_path+'/test_data.pt'
    scaler_path: str = preprocessed_data_path+'/scaler.pkl'
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
    csv_origin_path: str = original_data_path
    preprocessed_path: str = preprocessed_data_path
    window_meta_path: str = preprocessed_data_path+'/window_meta.pkl'

