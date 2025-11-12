from dataclasses import dataclass, field
from enum import Enum, auto
from glob import glob
from typing import List

import torch

# from utils import SNR_to_noise
import numpy as np
import pdb

def SNR_to_noise(snr):
    snr = 10 ** (snr / 10)
    noise_std = 1 / np.sqrt(2 * snr)

    return noise_std

# model을 실행할 때 사용하는 디바이스 설정
# CUDA가 사용 가능하면 GPU를, 그렇지 않으면 CPU를 사용
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" =========================== main.py 흐름 제어 변수 =========================== """

is_train_phase = False # 나중에 변경

# True면 mutual information model 도 같이 학습함
# is_learning_minet = True
is_learning_minet = False

# True면 이미 전처리된 데이터로 학습 및 평가 진행, False면 preprocess 실행
is_preprocessed = True
# is_preprocessed = False

# 이상치 제거를 진행하지 않게 만듦
# is_skip_outlier_eliminate = True
is_skip_outlier_eliminate = False

# 학습이 완료되었는지 여부 -> True면 학습된 모델로 평가, False면 학습 진행
# is_trained = True
is_trained = False

# 성능평가를 진행하지 않게 만듦
# is_skip_performance = True
is_skip_performance = False

# 최종 전체 복원을 진행하지 않게 만듦
is_skip_full_reconstruct = True
# is_skip_full_reconstruct = False

# voltage 0값 제거 진행하지 않게 만듦
# is_skip_0_in_voltage = True
is_skip_0_in_voltage = False

# data.make_EDA_images, reshape_csv 에서 eda 의 대상 타입을 지정하는 변수 (지정하지 않으면 reconstruction)
# eda_target_type = "original"
# eda_target_type = "outlier_cut"
eda_target_type = ""

# data.make_EDA_images 에서 row index를 x label로 생성함, False인 경우 Time축으로 생성
is_row_x_label_on_EDA = True
# is_row_x_label_on_EDA = False

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
case_index = "10019.1.1"

# 모델 종류
class ModelType(Enum):
    DEEPSC = "deepsc"
    LSTM = "lstm"
    GRU = "gru"
    ATTENTION_LSTM = "at_lstm"
    ONLY_CHANNEL = "only_channel"


# 손실함수 종류
class LossType(Enum):
    MSE = "MSE"
    MAE = "MAE"
    Huber = "Huber"


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
LOSS_TYPES = [loss.value for loss in LossType]  # ['MSE', 'MAE', 'Huber']
CHANNEL_TYPES = [
    channel.value for channel in ChannelType
]  # ['no_channel', 'AWGN', 'rayleigh', 'rician']
SCALER_TYPES = [scaler.value for scaler in ScalerType]  # ['minmax', 'zscore']

# 현재 사용할 변수 설정들
model_type = ModelType.DEEPSC.value  # default
loss_type = LossType.MSE.value  # MSE로 설정
channel_type = ChannelType.RAYLEIGH.value  # no_channel 선택
scaler_type = ScalerType.ZSCORE.value  # minmax 선택

# channel 노이즈 설정
snr_list = [3, 6, 9, 12, 15, 18]
noise_std = SNR_to_noise(snr_list[2])

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
target_length = 512  # 256, 512 , segment_leng과 동일
# target_length = 1024  # 256, 512
# exclude batteries
exclude_batteries = ["B0049", "B0050", "B0051", "B0052"]
# exclude_batteries = []
# exclude_batteries = ["B0029", "B0030", "B0031", "B0032"]
current_remove_groups = {
    "-3_group": (-3.5, -2.05),  # -3.5 <= Current_measured < -2.5
    "-4_group": (-3.95, -3.5),  # -4.0 < Current_measured < -3.5
    "-4_out_group" : (-4.05, -4.95)
}

segment_trimmed_dfs = {}

""" =========================== EDA용 변수 설정 =========================== """
# eda_readme_files = glob("./original_dataset/extra_infos/README_*.txt")
# eda_merged_path = f"./data/merged_recons/case_{case_index}/"
# eda_output_prefix_path = f"./data/EDA_images_recons/case_{case_index}/"
# eda_full_recon_path = f"./data/full_recons/case_{case_index}/"
eda_readme_files = glob("./original_dataset/extra_infos/README_*.txt")
# (f-string 변수 -> lambda 함수로 변경)
get_eda_merged_path = lambda: f"./data/merged_recons/case_{case_index}/"
get_eda_output_prefix_path = lambda: f"./data/EDA_images_recons/case_{case_index}/"
get_eda_full_recon_path = lambda: f"./data/full_recons/case_{case_index}/"

""" =========================== 경로 설정 =========================== """

# # 전처리 입력으로 사용할 데이터 경로 (merged)
# original_data_path = "original_dataset/data/"

# # 중간에 이상치 제거 버전 csv 저장할 경로 -> 나중에 이걸 최종 csv 복원 비교의 원본 csv으로 사용
# outlier_cut_csv_path = (
#     # f"./cycle_preprocess/csv/outlier_cut/"
#     f"./cycle_preprocess/csv/outlier_cut/threshold_{outlier_threshold}/cycle_len_{target_length}/"
# )

# # 중간에 resampled 된 버전 csv 저장할 경로 -> 확인용 -> 이제 preprocessed_csv와 동일해짐 -> 사용 x
# resampled_csv_folder = f"cycle_preprocess/csv/reshaped/resampled_{target_length}/"

# # 중간에 전처리 다 된 버전 csv 저장할 경로 -> 확인용
# preprocessed_csv_path = f"cycle_preprocess/csv/total_preprocessed/processed_{scaler_type}_{target_length}_threshold_{outlier_threshold}/case_{case_index}"

# # merged의 파일에서 이상치가 제거되며 전처리 된 데이터 경로 (train_data.pt, test_data.pt)
# preprocessed_data_path = f"./cycle_preprocess/total_preprocessed/processed_{scaler_type}_{target_length}/case_{(str(case_index)).split('.')[0]}/"
# # preprocessed_data_path = f"./cycle_preprocess/total_preprocessed/processed_{scaler_type}_{target_length}/case_{case_index}/"

# # 모델 저장 경로
# model_checkpoint_path = f"./checkpoints/case_{case_index}/{loss_type}/{model_type}/{model_type}_battery_epoch"
# # model_checkpoint_path = f"./checkpoints/case_58.1.1/{loss_type}/{model_type}/{model_type}_battery_epoch"

# # train 중 validation 복원 plot 저장 경로
# save_fig_dir = f"results/case{case_index}/{channel_type}_{model_type}_{loss_type}"

# # 복원 csv 저장 경로
# save_reconstruct_dir = f"reconstruction/case{case_index}/reconstructed_{channel_type}_{model_type}_{loss_type}"

# # 복원 성능 plot 저장 경로
# save_performance_dir = (
#     f"results/performance_test/case{case_index}/{channel_type}_{model_type}_{loss_type}"
# )
# 전처리 입력으로 사용할 데이터 경로 (merged)
original_data_path = "original_dataset/data/" # (이 변수는 정적이므로 그대로 둡니다)

# (f-string 변수 -> lambda 함수로 변경)
get_outlier_cut_csv_path = lambda: (
    f"./cycle_preprocess/csv/outlier_cut/threshold_{outlier_threshold}/cycle_len_{target_length}/"
)
get_resampled_csv_folder = lambda: f"cycle_preprocess/csv/reshaped/resampled_{target_length}/"

get_preprocessed_csv_path = lambda: (
    f"cycle_preprocess/csv/total_preprocessed/processed_{scaler_type}_{target_length}_threshold_{outlier_threshold}/case_{case_index}"
)
get_case_prefix = lambda: (str(case_index)).split('.')[0] # case_index에서 앞부분만 따는 헬퍼 함수
get_preprocessed_data_path = lambda: (
    f"./cycle_preprocess/total_preprocessed/processed_{scaler_type}_{target_length}/case_{get_case_prefix()}/"
)
get_model_checkpoint_path = lambda: (
    f"./checkpoints/case_{case_index}/{loss_type}/{model_type}/{model_type}_battery_epoch"
)
get_save_fig_dir = lambda: (
    f"results/case{case_index}/{channel_type}_{model_type}_{loss_type}"
)
get_save_reconstruct_dir = lambda: (
    f"reconstruction/case{case_index}/reconstructed_{channel_type}_{model_type}_{loss_type}"
)
get_save_performance_dir = lambda: (
    f"results/performance_test/case{case_index}/{channel_type}_{model_type}_{loss_type}"
)

""" ========================= 모델 파라미터 설정 ======================== """

# epochs
train_epochs = 120
# batch size
train_batch_size = 8
# learning rate
train_lr = 1e-5
# input dimension
input_dim = 5
# input_dim = 6
segment_length_n = 8  # 입력 시퀀스 길이를 segment_length_n 등분하여 처리

get_train_batch_size = lambda: train_batch_size
get_segment_length_n = lambda: segment_length_n

""" ========================= 각 기능 모듈별 파라미터 정리 ========================= """
# # preprocess
# @dataclass(unsafe_hash=True)
# class PreprocessParams:
#     scaler_type = scaler_type
#     target_length = target_length
#     exclude_batteries = exclude_batteries
#     original_data_path = original_data_path  # 입력 경로
#     resampled_csv_folder = resampled_csv_folder  # resampled 결과 csv 경로
#     outlier_cut_csv_path = outlier_cut_csv_path  # 이상치 제거 결과 csv 경로
#     preprocessed_csv_path = preprocessed_csv_path  # 전처리 다 된 csv 경로
#     preprocessed_data_path = preprocessed_data_path  # 전처리 다 된 .pt 경로
#     outlier_threshold = outlier_threshold
#     current_remove_groups = current_remove_groups


# # train
# @dataclass
# class TrainParams:
#     train_pt: str = preprocessed_data_path + "/train_data.pt"
#     validate_pt: str = preprocessed_data_path + "/val_data.pt"
#     scaler_path: str = preprocessed_data_path + "/scaler.pkl"
#     model_save_path: str = model_checkpoint_path
#     num_epochs: int = train_epochs
#     batch_size: int = train_batch_size
#     lr: float = train_lr
#     loss_type: str = loss_type
#     lambda_feat: float = 3  # (추가) 특징 공간 손실의 가중치


# # test
# @dataclass
# class TestParams:
#     loss_type: str = loss_type
#     model_type: str = model_type
#     channel_type: str = channel_type
#     csv_origin_path: str = outlier_cut_csv_path  # 복원본과 비교할 대상 (이상치제거본)
#     preprocessed_path: str = preprocessed_data_path
#     save_performance_dir = save_performance_dir
#     save_reconstruct_dir = save_reconstruct_dir
#     feature_cols: List[str] = field(default_factory=lambda: feature_cols.copy())
#     train_pt = preprocessed_data_path + "/train_data.pt"
#     test_pt = preprocessed_data_path + "/test_data.pt"
#     scaler_path = preprocessed_data_path + "/scaler.pkl"
#     target_length = target_length  # P2.4에 사용


# # full reconstruct
# @dataclass
# class ReconstructParams:
#     preprocessed_path: str = preprocessed_data_path
#     save_performance_dir = save_performance_dir
#     save_reconstruct_dir = eda_full_recon_path
#     csv_origin_path: str = outlier_cut_csv_path  # 원본 길이를 찾기 위한 목적도 존재
#     feature_cols: List[str] = field(default_factory=lambda: feature_cols.copy())
#     train_pt = preprocessed_data_path + "/train_data.pt"
#     val_pt = preprocessed_data_path + "/val_data.pt"
#     test_pt = preprocessed_data_path + "/test_data.pt"
#     scaler_path = preprocessed_data_path + "/scaler.pkl"
#     target_length = target_length  # P2.4에 사용

# preprocess
@dataclass(unsafe_hash=True)
class PreprocessParams:
    # (단순 변수 -> field(default_factory=...)로 변경)
    scaler_type: str = field(default_factory=lambda: scaler_type)
    target_length: int = field(default_factory=lambda: target_length)
    exclude_batteries: List[str] = field(default_factory=lambda: exclude_batteries)
    original_data_path: str = field(default=original_data_path) # 정적 변수는 default=
    resampled_csv_folder: str = field(default_factory=get_resampled_csv_folder)
    outlier_cut_csv_path: str = field(default_factory=get_outlier_cut_csv_path)
    preprocessed_csv_path: str = field(default_factory=get_preprocessed_csv_path)
    preprocessed_data_path: str = field(default_factory=get_preprocessed_data_path)
    outlier_threshold: int = field(default_factory=lambda: outlier_threshold)
    current_remove_groups: dict = field(default_factory=lambda: current_remove_groups)
    segment_length_n : int = field(default_factory=get_segment_length_n)

# train
@dataclass
class TrainParams:
    # (경로 변수 -> field(default_factory=...)로 변경)
    train_pt: str = field(default_factory=lambda: get_preprocessed_data_path() + "/train_data.pt")
    validate_pt: str = field(default_factory=lambda: get_preprocessed_data_path() + "/val_data.pt")
    scaler_path: str = field(default_factory=lambda: get_preprocessed_data_path() + "/scaler.pkl")
    model_save_path: str = field(default_factory=get_model_checkpoint_path)
    save_fig_dir:str = field(default_factory=get_save_fig_dir)

    # (단순 변수 -> field(default_factory=...)로 변경)
    num_epochs: int = field(default=train_epochs) # (이 변수들은 루프에서 안바뀌면 default= 그대로 둬도 됨)
    batch_size: int = field(default_factory=get_train_batch_size)
    lr: float = field(default=train_lr)
    loss_type: str = field(default_factory=lambda: loss_type)
    lambda_feat: float = 3  # (정적 값이면 그대로 둠)
    segment_length_n : int = field(default_factory=get_segment_length_n)

# test
@dataclass
class TestParams:
    # (단순 변수 -> field(default_factory=...)로 변경)
    loss_type: str = field(default_factory=lambda: loss_type)
    model_type: str = field(default_factory=lambda: model_type)
    channel_type: str = field(default_factory=lambda: channel_type)

    # (경로 변수 -> field(default_factory=...)로 변경)
    csv_origin_path: str = field(default_factory=get_outlier_cut_csv_path)
    preprocessed_path: str = field(default_factory=get_preprocessed_data_path)
    save_performance_dir: str = field(default_factory=get_save_performance_dir)
    save_reconstruct_dir: str = field(default_factory=get_save_reconstruct_dir)

    feature_cols: List[str] = field(default_factory=lambda: feature_cols.copy())
    train_pt: str = field(default_factory=lambda: get_preprocessed_data_path() + "/train_data.pt")
    test_pt: str = field(default_factory=lambda: get_preprocessed_data_path() + "/test_data.pt")
    scaler_path: str = field(default_factory=lambda: get_preprocessed_data_path() + "/scaler.pkl")
    target_length: int = field(default_factory=lambda: target_length)
    segment_length_n : int = field(default_factory=get_segment_length_n)


# full reconstruct
@dataclass
class ReconstructParams:
    # (경로 변수 -> field(default_factory=...)로 변경)
    preprocessed_path: str = field(default_factory=get_preprocessed_data_path)
    save_performance_dir: str = field(default_factory=get_save_performance_dir) # (get_save_performance_dir 사용)
    save_reconstruct_dir: str = field(default_factory=get_eda_full_recon_path)
    csv_origin_path: str = field(default_factory=get_outlier_cut_csv_path)

    feature_cols: List[str] = field(default_factory=lambda: feature_cols.copy())
    train_pt: str = field(default_factory=lambda: get_preprocessed_data_path() + "/train_data.pt")
    val_pt: str = field(default_factory=lambda: get_preprocessed_data_path() + "/val_data.pt")
    test_pt: str = field(default_factory=lambda: get_preprocessed_data_path() + "/test_data.pt")
    scaler_path: str = field(default_factory=lambda: get_preprocessed_data_path() + "/scaler.pkl")
    target_length: int = field(default_factory=lambda: target_length)
