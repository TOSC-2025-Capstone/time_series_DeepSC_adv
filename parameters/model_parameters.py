import json
import pickle

## 방법 1 : 각 변수 그대로 import * 하기 -> 나중에 파라미터 셋을 변경하면서 for 돌리기는 어려움
# LSTM, GRU, AT_LSTM 공통
input_dim = 6
hidden_dim = 128
compressed_len = 64
compressed_features = 3
num_layers = 2
dropout = 0.1
reconstruct_len = 128
reconstruct_features = 6
seq_len = 128

# DeepSC 관련
dff = 512
num_heads = 4
d_model = 128
max_len = 128
d_comp = 3

## 방법 2 : 딕셔너리를 한번에 전달 -> 딕셔너리를 한번에 여러개 만들어 두면 방법 1의 단점 해소 가능
## 대신 모델 init 구조에서도 **params 추가, 인스턴스 변수에서 딕셔너리 받은거 구조분해할당 해야함
model_params = {
    # lstm, gru, at_lstm
    "input_dim": 6,
    "hidden_dim": 128,
    "compressed_len": 32,
    "compressed_features": 6,
    "num_layers": 2,
    "dropout": 0.1,
    "reconstruct_len": 128,
    "reconstruct_features": 6,
    "seq_len": 128,

    # deepsc 
    "dff": 512,
    "num_heads": 4,
    "d_model": 128,
    "max_len": 128, 
    "d_comp": 3
}

# num_layers=4,
# input_dim=input_dim,
# max_len=window_size,
# d_model=128,
# num_heads=8,
# dff=512,
# dropout=0.1,
# compressed_len=64

## 방법 3
## config.json 파일에서 파라미터 불러오기
# with open("config.json", "r") as f:
#     model_params = json.load(f)

## 모델에 파라미터 전달
# model = LSTMDeepSC(**model_params)
