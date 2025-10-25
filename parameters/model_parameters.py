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
    # "compressed_len": 64,  # 모든 모델의 압축된 시퀀스 길이
    "num_layers": 4, # 모든 모델의 레이어 수
    "dropout": 0.1,
    # lstm, gru, at_lstm
    "hidden_dim": 512,  # lstm, gru, at_lstm hidden dim
    "seq_len": 512,  # lstm, gru, at_lstm 입력 시퀀스 길이
    # deepsc
    "dff": 512,  # deepsc 모델 최대 노드 수
    "num_heads": 4,  # deepsc 모델 헤드 수
    "d_model": 512,  # deepsc 모델 피쳐 확장 차원
    "max_len": 512,  # deepsc 모델 출력 sequence length
    "d_comp": 3,  # deepsc 압축 피쳐 수
    # "d_comp": 1,  # deepsc 압축 피쳐 수

    # 10.04 addition
    "use_itransformer" : True,  # True=iTransformer, False=Transformer

    # 10.12 snr_db parameter
    "snr_db": 5,  # AWGN 채널 SNR 값
    # "snr_db": 5,  # AWGN 채널 SNR 값, noise_type_1
    # "snr_db": 10,  # AWGN 채널 SNR 값, noise_type_2
    # "snr_db": 15,  # AWGN 채널 SNR 값, noise_type_3
    # "snr_db": 20,  # AWGN 채널 SNR 값, noise_type_4

    "kernel_size": 3,
}

""" 주의사항
    iTransformer의 window_size는 변수 개수 기준입니다
    왠만하면 transformer에서만 윈도우를 적용하기

    배터리 데이터: 5개 피처 (전압, 전류, 온도 등)
    window_size=3: 각 변수가 인접 3개 변수만 참조
    window_size=5: 모든 변수 참조 (일반 Attention과 동일)

    Transformer의 window_size는 시간 스텝 기준입니다:

    window_size=128: 각 시점이 앞뒤 64 스텝만 참조 (권장값 = 입력 길이의 1/4)
    window_size=512: 전체 시퀀스 참조
"""
