import json
import pickle

model_params = {
    # 공통
    "input_dim": 5,  # 모든 모델의 입력 피쳐 수
    # 여기
    # "compressed_len": 512,  # 모든 모델의 압축된 시퀀스 길이
    # "compressed_len": 384,  # 모든 모델의 압축된 시퀀스 길이
    # "compressed_len": 256,  # 모든 모델의 압축된 시퀀스 길이
    "compressed_len": 128,  # 모든 모델의 압축된 시퀀스 길이
    "num_layers": 2, # 모든 모델의 레이어 수
    "dropout": 0.1,
    # lstm, gru, at_lstm
    "hidden_dim": 512,  # lstm, gru, at_lstm hidden dim
    "compressed_features": 3,  # lstm, gru 압축 피쳐 수
    "reconstruct_len": 512,  # lstm, gru, at_lstm 모델 출력 sequence length
    "seq_len": 512,  # lstm, gru, at_lstm 입력 시퀀스 길이
    # "reconstruct_features": 3,  # lstm, gru, at_lstm 복원 피쳐 수
    # deepsc
    "dff": 512,  # deepsc 모델 최대 노드 수
    "num_heads": 4,  # deepsc 모델 헤드 수
    "d_model": 512,  # deepsc 모델 입력 sequence length
    "max_len": 512,  # deepsc 모델 출력 sequence length
    "d_comp": 3,  # deepsc 압축 피쳐 수
}
