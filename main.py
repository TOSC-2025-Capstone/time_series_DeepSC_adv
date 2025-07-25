# -*- coding: utf-8 -*-
import os
import argparse
import time
import json

# from pandas.compat.pyarrow import pa
import torch
import random
import torch.nn as nn
import torch
import numpy as np
from models.transceiver import DeepSC
from models.mutual_info import Mine
from torch.utils.data import DataLoader
from tqdm import tqdm
import pdb

from models.lstm import LSTMDeepSC
from models.gru import GRUDeepSC
from models.attention_lstm import LSTMAttentionDeepSC
from models.transceiver import DeepSC

from preprocess import load_all_valid_csv_tensors
from cycle_preprocess.cycle_preprocess import cycle_preprocess
from train import train_model
from performance import reconstruct_battery_series

# from performance_cycle import reconstruct_battery_series2

# 기타 매개변수, 모델 파라미터 모두 가져오기
from parameters.parameters import *
from parameters.model_parameters import *

preprocess_params = PreprocessParams()

parser = argparse.ArgumentParser()
# parser.add_argument('--data-dir', default='data/train_data.pkl', type=str)

print(torch.__version__)  # PyTorch 버전 확인
print(torch.version.cuda)  # PyTorch에서 사용하는 CUDA 버전 확인
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device, "현재 사용중인 디바이스")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    setup_seed(42)
    if is_first == True:
        cycle_preprocess()

    # model create
    model = None
    if model_type == "deepsc":
        model = DeepSC(params=model_params).to(device)
        # 아래와 같이 개별 변수를 정의하는 것도 가능은 함, model_parameters에서 방법 1 주석
        # model = DeepSC(
        #     num_layers=num_layers,
        #     input_dim=input_dim,
        #     max_len=max_len,
        #     d_model=d_model,
        #     num_heads=num_heads,
        #     dff=dff,
        #     dropout=dropout,
        #     compressed_len=compressed_len
        # )
    elif model_type == "lstm":
        model = LSTMDeepSC(params=model_params).to(device)
    elif model_type == "gru":
        model = GRUDeepSC(params=model_params).to(device)
    elif model_type == "at_lstm":
        model = LSTMAttentionDeepSC(params=model_params).to(device)

    # train
    train_model(model=model, device=device)

    # pdb.set_trace()
    # test + result figuring
    # reconstruct_battery_series(model=model, device=device)
