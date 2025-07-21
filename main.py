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
from train import train_model
from performance import reconstruct_battery_series

# 기타 매개변수, 모델 파라미터 모두 가져오기
from parameters.parameters import *
from parameters.model_parameters import *

preprocess_params = PreprocessParams()

parser = argparse.ArgumentParser()
#parser.add_argument('--data-dir', default='data/train_data.pkl', type=str)

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

if __name__ == '__main__':
    # setup_seed(10)

    """ define optimizer and loss function """
    # preprocess
    if is_first == True:
        load_all_valid_csv_tensors(
            folder_path=preprocess_params.folder_path,
            feature_cols=preprocess_params.feature_cols,
            batch_size=preprocess_params.batch_size,
            save_split_path=preprocess_params.save_split_path,
            split_ratio=preprocess_params.split_ratio,
            window_size=preprocess_params.window_size,
            stride=preprocess_params.stride,
        )

    # model create
    model = None
    # pdb.set_trace()
    if model_type == 'deepsc':
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
    elif model_type == 'lstm':
        model = LSTMDeepSC(params=model_params).to(device)
    elif model_type == 'gru' :
        model = GRUDeepSC(params=model_params).to(device)
    elif model_type == 'at_lstm':
        model = LSTMAttentionDeepSC(params=model_params).to(device)

    # train
    train_model(model=model, device=device)

    # test + result figuring
    reconstruct_battery_series(model=model, device=device)

    
