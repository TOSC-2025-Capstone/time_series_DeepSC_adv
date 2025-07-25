import os
import pandas as pd
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt


def reverse_resample(compressed_df, original_length):
    """
    256 길이의 압축된 데이터를 원래 길이로 복원
    """
    # 현재 인덱스(256개)를 시간 축으로 사용
    current_indices = np.arange(len(compressed_df))
    # 목표 인덱스 생성 (원래 길이)
    target_indices = np.linspace(0, len(compressed_df) - 1, original_length)

    restored_data = {}
    for column in compressed_df.columns:
        if column != "cycle_idx":  # cycle_idx는 제외
            # 1D 보간 함수 생성
            f = interpolate.interp1d(current_indices, compressed_df[column].values)
            # 새로운 인덱스에 대한 값 계산
            restored_data[column] = f(target_indices)

    # cycle_idx는 원래 값 복제
    if "cycle_idx" in compressed_df.columns:
        restored_data["cycle_idx"] = [
            compressed_df["cycle_idx"].iloc[0]
        ] * original_length

    return pd.DataFrame(restored_data)
