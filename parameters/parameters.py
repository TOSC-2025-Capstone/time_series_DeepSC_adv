from dataclasses import dataclass, field
from models.lstm import LSTMDeepSC
from models.gru import GRUDeepSC
from models.attention_lstm import LSTMAttentionDeepSC
from models.transceiver import DeepSC

import torch
from typing import List
from enum import Enum, auto

# model을 실행할 때 사용하는 디바이스 설정
# CUDA가 사용 가능하면 GPU를, 그렇지 않으면 CPU를 사용
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" =========================== main.py 흐름 제어 변수 =========================== """

# 처음 실행 여부 -> True면 이미 전처리된 데이터로 학습 및 평가 진행, False면 preprocess 실행
# is_preprocessed = True
is_preprocessed = False

# 학습이 완료되었는지 여부 -> True면 학습된 모델로 평가, False면 학습 진행
# is_trained = True
is_trained = False

"""  =========================== 모델 파라미터 설정 =========================== """
"""
    자신이 사용하는 모델에 따라 아래처럼 작성
        DeepSC => models_type_arr[0], case_index는 8.1.x 로 작성 (8.1.0, 8.1.1, ...) 
        LSTMDeepSC => models_type_arr[1], case_index는 8.2.x 로 작성 (8.2.0, 8.2.1, ...)
        GRUDeepSC => models_type_arr[2], case_index는 8.3.x 로 작성 (8.3.0, 8.3.1, ...)

    train.py의 figure, performance_test.py의 figure, 복원 csv 저장 경로 등에서 이 case_index, loss_type, model_type 들을 사용하므로
    테스트할 때는 이 부분을 자신의 버전으로 적용했는 지 반드시 잘 보고 실행해야합니다! (다른 테스트 결과를 오염시킬 수 있음)
"""
# 테스트 케이스 인덱스
case_index = 9.1


# 모델 종류
class ModelType(Enum):
    DEEPSC = "deepsc"
    LSTM = "lstm"
    GRU = "gru"
    ATTENTION_LSTM = "at_lstm"


# 손실함수 종류
class LossType(Enum):
    MSE = "MSE"
    MAE = "MAE"
    SMOOTH_L1 = "SmoothL1Loss"


# 채널 타입
class ChannelType(Enum):
    NO_CHANNEL = "no_channel"
    AWGN = "AWGN"
    RAYLEIGH = "rayleigh"
    RICIAN = "rician"


# 스케일러 타입
class ScalerType(Enum):
    MINMAX = "minmax"
    ZSCORE = "zscore"


# 각 타입의 모든 값을 얻기 위한 리스트 (사용 안해도 됨)
MODEL_TYPES = [
    model.value for model in ModelType
]  # ['deepsc', 'lstm', 'gru', 'at_lstm']
LOSS_TYPES = [loss.value for loss in LossType]  # ['MSE', 'MAE', 'SmoothL1Loss']
CHANNEL_TYPES = [
    channel.value for channel in ChannelType
]  # ['no_channel', 'AWGN', 'rayleigh', 'rician']
SCALER_TYPES = [scaler.value for scaler in ScalerType]  # ['minmax', 'zscore']

# 현재 사용할 설정들
model_type = ModelType.DEEPSC.value  # GRU 선택
loss_type = LossType.MSE.value  # MSE로 설정
channel_type = ChannelType.NO_CHANNEL.value  # no_channel 선택
scaler_type = ScalerType.MINMAX.value  # minmax 선택

# feature cols (inputs)
feature_cols = [
    "Voltage_measured",
    "Current_measured",
    "Temperature_measured",
    "Current_load",
    "Voltage_load",
    "Time",
]

# outlier elimination threshold
outlier_threshold = 7  # 3, 5, 7, 10
# cycle preprocess length
target_length = 256  # 256, 512
# exclude batteries
exclude_batteries = ["B0049", "B0050", "B0051", "B0052"]

""" =========================== 경로 설정 =========================== """

# 전처리 입력으로 사용할 데이터 경로 (merged)
original_data_path = "original_dataset/data/"

# 중간에 이상치 제거 버전 csv 저장할 경로 -> 나중에 이걸 최종 csv 복원 비교의 원본 csv으로 사용
outlier_cut_csv_path = (
    # f"./cycle_preprocess/csv/outlier_cut/"
    f"./cycle_preprocess/csv/outlier_cut/threshold_{outlier_threshold}/"
)

# 중간에 resampled 된 버전 csv 저장할 경로 -> 확인용 -> 이제 preprocessed_csv와 동일해짐 -> 사용 x
resampled_csv_folder = f"cycle_preprocess/csv/reshaped/resampled_{target_length}/"

# 중간에 전처리 다 된 버전 csv 저장할 경로 -> 확인용
preprocessed_csv_path = (
    f"cycle_preprocess/csv/total_preprocessed/processed_{scaler_type}_{target_length}/"
)

# merged의 파일에서 이상치가 제거되며 전처리 된 데이터 경로 (train_data.pt, test_data.pt)
preprocessed_data_path = (
    f"./cycle_preprocess/total_preprocessed/processed_{scaler_type}_{target_length}/"
)

# 모델 저장 경로
model_checkpoint_path = f"./checkpoints/case_{case_index}/{loss_type}/{model_type}/{model_type}_battery_epoch"


# train 중 validation 복원 plot 저장 경로
save_fig_dir = f"results/case{case_index}/{channel_type}_{model_type}_{loss_type}"

# 복원 csv 저장 경로
save_reconstruct_dir = f"reconstruction/case{case_index}/reconstructed_{channel_type}_{model_type}_{loss_type}"

# 복원 성능 plot 저장 경로
save_performance_dir = (
    f"results/performance_test/case{case_index}/{channel_type}_{model_type}_{loss_type}"
)

""" ========================= 모델 파라미터 설정 ======================== """

# epochs
train_epochs = 80
# batch size
train_batch_size = 32
# learning rate
tratin_lr = 1e-5
# input dimension
input_dim = 6


""" ========================= 각 기능 모듈별 파라미터 정리 ========================= """


# preprocess
@dataclass(unsafe_hash=True)
class PreprocessParams:
    scaler_type = scaler_type
    target_length = target_length
    exclude_batteries = exclude_batteries
    original_data_path = original_data_path  # 입력 경로
    resampled_csv_folder = resampled_csv_folder  # resampled 결과 csv 경로
    outlier_cut_csv_path = outlier_cut_csv_path  # 이상치 제거 결과 csv 경로
    preprocessed_csv_path = preprocessed_csv_path  # 전처리 다 된 csv 경로
    preprocessed_data_path = preprocessed_data_path  # 전처리 다 된 .pt 경로
    outlier_threshold = outlier_threshold


# train
@dataclass
class TrainParams:
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
