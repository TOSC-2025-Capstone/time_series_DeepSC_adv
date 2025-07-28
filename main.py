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

from cycle_preprocess.cycle_preprocess import cycle_preprocess
from train import train_model

from performance_cycle import performance_cycle

# 기타 매개변수, 모델 파라미터 모두 가져오기
from parameters.parameters import *
from parameters.model_parameters import *

# 파라미터 클래스 가져오기
# preprocess_params = PreprocessParams()
train_params = TrainParams()
test_params = TestParams()

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

    print("========================== preprocess ==========================\n")
    if is_preprocessed == False:
        cycle_preprocess(scaler_type=scaler_type)
        print("사이클 전처리가 완료되었습니다.")
    else:
        print("사이클 전처리가 이미 완료되었습니다. 기존 데이터를 사용합니다.")

    # model create
    print("========================== model_select ==========================\n")
    model = None
    if model_type == "deepsc":
        model = DeepSC(params=model_params).to(device)
        print("DeepSC 모델이 선택되었습니다.")
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
        print("LSTMDeepSC 모델이 선택되었습니다.")
    elif model_type == "gru":
        model = GRUDeepSC(params=model_params).to(device)
        print("GRUDeepSC 모델이 선택되었습니다.")
    elif model_type == "at_lstm":
        model = LSTMAttentionDeepSC(params=model_params).to(device)
        print("LSTMAttentionDeepSC 모델이 선택되었습니다.")

    # train
    if is_trained == False:
        print("========================== train ==========================\n")
        model.train()
        if model.training:
            print("현재 모델은 training 모드입니다.")
        else:
            print("현재 모델은 evaluation (eval) 모드입니다.")
        train_model(params=train_params, model=model, device=device)

    print(
        "========================== best checkpoint load ==========================\n"
    )
    try:
        if os.path.exists(model_checkpoint_path):
            model.load_state_dict(
                torch.load(f"{model_checkpoint_path}best.pth", map_location=device)
            )
            print("모델이 성공적으로 로드되었습니다.")
    except Exception as e:
        print(f"모델 로드 실패: {e}")

    # test + result figuring
    pdb.set_trace()
    print("========================== test ==========================\n")
    model.eval()
    if model.training:
        print("현재 모델은 training 모드입니다.")
    else:
        print("현재 모델은 evaluation (eval) 모드입니다.")
    performance_cycle(params=test_params, model=model, device=device)
