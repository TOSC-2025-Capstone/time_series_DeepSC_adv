# -*- coding: utf-8 -*-
import argparse
import json
import os
import pdb
import random
import time

import gc
import torch

import numpy as np

# from pandas.compat.pyarrow import pa
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from cycle_preprocess.cycle_preprocess import cycle_preprocess
from models.mutual_info import Mine
from models.transceiver import DeepSC
from models.only_channel import OnlyChannel
from parameters.model_parameters import *

# 기타 매개변수, 모델 파라미터 모두 가져오기
from parameters.parameters import *
import parameters.parameters as p
from performance_cycle import performance_cycle
from train import train_model

from models.mutual_info import Mine

from utils import train_mi

parser = argparse.ArgumentParser()
# parser.add_argument('--data-dir', default='data/train_data.pkl', type=str)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

snr_list = [3, 6, 9, 12, 15, 18, 21]
seed_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
model_type_list = ["deepsc", "lstm"]
batch_size_list = [1, 2, 4, 8, 16]
base_case_number = int(case_index.split(".")[0])
model_type_index = 1 if model_type == "deepsc" else 2
# seq_len_list = [8, 16, 32, 64, 128, 256, 512]
seq_len_list = [8]

if __name__ == "__main__":
    print(torch.__version__)  # PyTorch 버전 확인
    print(torch.version.cuda)  # PyTorch에서 사용하는 CUDA 버전 확인
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device, "현재 사용중인 디바이스")

    seed = 1
    setup_seed(seed)
    # for batch_idx, batch_size in enumerate(batch_size_list):
    for model_type_index, v_model_type in enumerate(model_type_list):
        p.model_type=v_model_type
        if p.model_type == "deepsc":
            model_params["hidden_dim"] = 512
        else :
            model_params["hidden_dim"] = 128

        for snr_idx, v_snr_db in enumerate(snr_list):
            model_params["snr_db"] = v_snr_db
            for seq_len_idx, v_seq_len in enumerate(seq_len_list):
                case_number = base_case_number + seq_len_idx
                model_params["seq_len"] = v_seq_len
                model_params["max_len"] = v_seq_len
                model_params["compressed_len"] = v_seq_len // 4  # 압축 길이는 입력 길이의 1/4로 설정
                p.segment_length_n = v_seq_len

                checkpoint_index = f"{case_number}.{model_type_index+1}.1"
                # test_model_checkpoint_path = f"./checkpoints/case_{p.case_index}/{loss_type}/{model_type}/{model_type}_battery_epoch"
                test_model_checkpoint_path = f"./checkpoints/case_{checkpoint_index}/{loss_type}/{p.model_type}/{p.model_type}_battery_epoch"
                p.case_index = f"{case_number}.{model_type_index+1}.{snr_idx+2}"

                    # p.case_index = f"{int(case_number)}.{model_type_index}.1"  # case index 설정
                    # for snr_idx, snr in enumerate(snr_list):
                        # model_params["snr_db"] = snr
                        # p.case_index = f"{int(case_number)}.{model_type_index}.{snr_idx+2}"  # case index 설정

                # 파라미터 클래스 가져오기
                preprocess_params = PreprocessParams()
                train_params = TrainParams()
                test_params = TestParams()
                recons_params = ReconstructParams()

                test1 = model_params['snr_db']
                test2 = model_params["compressed_len"]

                print(f"========================== case_{p.case_index} start ==========================\n")
                print(f"model_type={p.model_type}, snr_db={test1}, compressed_len={test2}, caseindex: {p.case_index}, batch_size={p.train_batch_size}, test_path={test_model_checkpoint_path} 으로 설정되었습니다.")

                print("========================== preprocess ==========================\n")
                if is_preprocessed == False:
                    cycle_preprocess(preprocess_params=preprocess_params)
                    print("사이클 전처리가 완료되었습니다.")
                else:
                    print("사이클 전처리가 이미 완료되었습니다. 기존 데이터를 사용합니다.")

                # model create
                print("========================== model_select ==========================\n")
                model = None
                expert_model = None
                mi_net = None
                if p.model_type == "deepsc":
                    model = DeepSC(params=model_params, model_type=p.model_type).to(device)
                    print("Transformer 모델이 선택되었습니다.")
                    if is_learning_minet == True and channel_type != "no_channel":
                        mi_net = Mine().to(device)
                elif p.model_type == "lstm":
                    model = DeepSC(params=model_params, model_type=p.model_type).to(device)
                    print("LSTM_SC 모델이 선택되었습니다.")
                elif p.model_type == "gru":
                    model = DeepSC(params=model_params, model_type=p.model_type).to(device)
                    print("GRU_SC 모델이 선택되었습니다.")

                total, trainable = count_parameters(model)
                print(f"Total params: {total:,}")
                print(f"Trainable params: {trainable:,}")

                # train
                if p.is_trained == False:
                    print("========================== train ==========================\n")
                    model.train()
                    if model.training:
                        print("현재 모델은 training 모드입니다.")
                        # train_model(params=train_params, model=model, expert_model=expert_model, device=device, mi_net=mi_net)
                        train_model(params=train_params, model=model, device=device, mi_net=mi_net)
                    else:
                        print("현재 모델은 evaluation (eval) 모드입니다. 다시 실행하여 주세요")
                        exit(1)

                print(
                    "========================== best checkpoint load ==========================\n"
                )
                if p.model_type != "only_channel":
                    try:
                        if os.path.exists(test_model_checkpoint_path):
                            model.load_state_dict(
                                torch.load(f"{test_model_checkpoint_path}best.pth", map_location=device)
                            )
                            print("모델이 성공적으로 로드되었습니다.")
                    except Exception as e:
                        print(f"모델 로드 실패: {e}")

                # performance + result figuring
                print("========================== performance ==========================\n")
                p.is_train_phase = False
                if is_skip_performance == False:
                    model.eval()
                    if model.training:
                        print("현재 모델은 training 모드입니다.")
                    else:
                        print("현재 모델은 evaluation (eval) 모드입니다.")
                    performance_cycle(params=test_params, model=model, device=device, is_full_reconstruct=False)

                print("========================== full reconstruction ==========================\n")

                gc.collect()
                torch.cuda.empty_cache()
                if is_skip_full_reconstruct == False :
                    model.eval()
                    if model.training:
                        print("현재 모델은 training 모드입니다.")
                    else:
                        print("현재 모델은 evaluation (eval) 모드입니다.")
                    performance_cycle(
                        params=recons_params, model=model, device=device, is_full_reconstruct=True
                    )
