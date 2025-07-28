import json
import pickle

# ## 방법 1 : 각 변수 그대로 import * 하기 -> 나중에 파라미터 셋을 변경하면서 for 돌리기는 어려움
# # LSTM, GRU, AT_LSTM 공통
# input_dim = 6
# hidden_dim = 128
# compressed_len = 64
# compressed_features = 3
# num_layers = 2
# dropout = 0.1
# reconstruct_len = 128
# reconstruct_features = 6
# seq_len = 128

# # DeepSC 관련
# dff = 512
# num_heads = 4
# d_model = 128
# max_len = 128
# d_comp = 3

## 방법 2 : 딕셔너리를 한번에 전달 -> 딕셔너리를 한번에 여러개 만들어 두면 방법 1의 단점 해소 가능
## 대신 모델 init 구조에서도 **params 추가, 인스턴스 변수에서 딕셔너리 받은거 구조분해할당 해야함

# default
# model_params = {
#     # lstm, gru, at_lstm
#     "input_dim": 6,
#     "hidden_dim": 128,
#     "compressed_len": 64,
#     "compressed_features": 3,
#     "num_layers": 2,
#     "dropout": 0.1,
#     "reconstruct_len": 128,
#     "reconstruct_features": 6,
#     "seq_len": 128,

#     # deepsc
#     "dff": 512,
#     "num_heads": 2,
#     "d_model": 128,
#     "max_len": 128,
#     "d_comp": 6
# }

# case 6
model_params = {
    # 공통
    "input_dim": 6,  # 모든 모델의 입력 피쳐 수
    "compressed_len": 64,  # 모든 모델의 압축된 시퀀스 길이
    "num_layers": 4,  # 모든 모델의 레이어 수
    "dropout": 0.1,
    # lstm, gru, at_lstm
    "hidden_dim": 512,  # lstm, gru, at_lstm hidden dim
    "compressed_features": 6,  # lstm, gru 압축 피쳐 수
    "reconstruct_len": 256,  # lstm, gru, at_lstm 모델 입/출력 sequence length
    "reconstruct_features": 6,  # lstm, gru, at_lstm 복원 피쳐 수
    "seq_len": 256,  # lstm, gru, at_lstm 입력 시퀀스 길이
    # deepsc
    "dff": 512,  # deepsc 모델 최대 노드 수
    "num_heads": 4,  # deepsc 모델 헤드 수
    "d_model": 256,  # deepsc 모델 입력 sequence length
    "max_len": 256,  # deepsc 모델 출력 sequence length
    "d_comp": 6,  # deepsc 압축 피쳐 수
}

## 방법 3
## config.json 파일에서 파라미터 불러오기
# with open("config.json", "r") as f:
#     model_params = json.load(f)

## 모델에 파라미터 전달
# model = LSTMDeepSC(**model_params)
