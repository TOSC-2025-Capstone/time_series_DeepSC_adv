import os
import pandas as pd
import numpy as np
from scipy import interpolate
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
from torch.utils.data import TensorDataset
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import pdb
from methods import *

"""
# prepare_dataset.py

# prepare_dataset 함수:
모든 resampled discharge 데이터를 하나의 DataFrame으로 통합
이상치 처리 및 보간 수행
선택한 방식으로 스케일링
파일 단위로 8:2 비율의 train/test 분할
결과를 pt 파일과 scaler.pkl로 저장

# 처리 순서 (상세 내용은 methods.py 참고)
1. P1의 통합된 데이터프레임을 사용하여 피쳐 별 스케일 정규화를 진행
2. 정규화를 마친 통합 데이터프레임을 파일 단위로 분리 후
3. 각 파일을 256개 샘플로 리샘플링
4. 각 샘플들을 텐서로 변환하여 학습/테스트 데이터셋을 생성
5. 각 파일을 pickle, pt로 저장

# 정규화 스케일러
Voltage, Current, Temperature 등 각 특성이 서로 다른 scale을 가지고 있더라도, 각각 독립적으로 정규화되어 처리됩니다.

# TensorDataset 관련:
torch.utils.data import TensorDataset 추가
단순 텐서 대신 TensorDataset으로 변환하여 저장
향후 레이블이나 메타데이터 추가를 위한 확장성 확보

# 유틸리티 함수들:
load_and_filter_metadata(): 메타데이터 로드 및 배터리 필터링
load_resampled_data(): 리샘플링된 데이터 로드 및 통합
visualize_data_comparison(): 데이터 비교 시각화
scale_data(): 데이터 스케일링
split_and_transform_data(): 데이터 분할 및 텐서 변환

# 메인 함수:
prepare_dataset(): 전체 데이터 처리 파이프라인 실행
"""

# 한글깨짐 방지
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def prepare_dataset(df_cleaned, scaler_type="minmax", test_ratio=0.2):
    # 1. P1의 통합된 데이터프레임을 사용하여 피쳐 별 스케일 정규화를 진행
    df_cleaned = df_cleaned.drop(columns=["cycle_idx"], errors="ignore")
    if scaler_type == "minmax":
        scaler = MinMaxScaler()
    elif scaler_type == "zscore":
        scaler = StandardScaler()
    else:
        raise ValueError(
            "지원하지 않는 스케일러 타입입니다. 'minmax' 또는 'zscore'를 선택하세요."
        )
    scaled_df = scaler.fit_transform(df_cleaned)
    scaled_df = pd.DataFrame(scaled_df, columns=df_cleaned.columns)

    # 2. 정규화를 마친 통합 데이터프레임을 파일 단위로 분리 후
    grouped_dfs = grouping_df(df_cleaned)
    # grouped_dfs 리스트인지 하나하나 요소 찍어보고 for문 돌려서 3번 수행

    # 3. 각 파일을 256개 샘플로 리샘플링
    # for
    # resample_to_fixed_length

    # 4. 각 샘플들을 텐서로 변환하여 학습/테스트 데이터셋을 생성

    # 5. 각 파일을 pickle, pt로 저장
    return


if __name__ == "__main__":
    print("처리 순서:")
    print("1. outlier_eliminate.py를 실행하여 out_cut 데이터 생성")
    print("2. cycle_reshape.py를 실행하여 resampled 데이터 생성")
    print("3. prepare_dataset.py를 실행하여 전처리된 데이터셋 생성")
    print("4. preprocessed_data_check.py를 실행하여 결과 비교")

    # MinMax 스케일링으로 데이터셋 준비
    print("\n=== MinMax Scaling ===")
    train_data_minmax, test_data_minmax, scaler_minmax = prepare_dataset(
        scaler_type="minmax"
    )

    if train_data_minmax is None:
        print("MinMax 스케일링 처리 실패. 프로그램을 종료합니다.")
        exit()

    # Z-score 스케일링으로 데이터셋 준비
    print("\n=== Z-score Scaling ===")
    train_data_zscore, test_data_zscore, scaler_zscore = prepare_dataset(
        scaler_type="zscore"
    )

    if train_data_zscore is None:
        print("Z-score 스케일링 처리 실패. 프로그램을 종료합니다.")
        exit()

    print("\n모든 처리가 완료되었습니다!")
    print("이제 preprocessed_data_check.py를 실행하여 결과를 비교해보세요.")
