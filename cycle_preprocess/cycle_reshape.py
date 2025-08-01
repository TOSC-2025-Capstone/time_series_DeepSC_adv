import os
import pandas as pd
import numpy as np
from scipy import interpolate
import pdb

"""
# cycle_reshape.py

resample_to_fixed_length 함수:

입력: 데이터프레임, 목표 길이(기본값 256)
각 컬럼(cycle_idx 제외)을 선형 보간(linear interpolation)을 사용하여 256개 샘플로 리샘플링
cycle_idx는 원래 값 유지
"""


# P2.3
def resample_to_fixed_length(df, target_length=256, resampled_output_folder=None):
    """
    데이터프레임의 각 컬럼을 target_length 개수로 리샘플링
    """
    # 현재 인덱스를 시간 축으로 사용
    current_indices = np.arange(len(df))
    # 목표 인덱스 생성 (균일한 간격)
    target_indices = np.linspace(0, len(df) - 1, target_length)

    resampled_data = {}
    for column in df.columns:
        if column != "cycle_idx":  # cycle_idx는 제외
            # 1D 보간 함수 생성
            f = interpolate.interp1d(current_indices, df[column].values)
            # 새로운 인덱스에 대한 값 계산
            resampled_data[column] = f(target_indices)

            # 결과 저장 (이제는 total_preprocessed와 동일한 csv라 주석 처리)
            # output_path = os.path.join(resampled_output_folder, df["cycle_idx"])
            # resampled_data[column].to_csv(output_path, index=False)
            # print(f"완료: {df["cycle_idx"]} ({len(df)} -> {target_length} samples)")

    return pd.DataFrame(resampled_data)


# if __name__ == "__main__":
