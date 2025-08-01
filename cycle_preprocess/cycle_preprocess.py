import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset
import pickle
from .methods import *
from .outlier_eliminate import process_and_save_outlier_data
from .cycle_reshape import (
    resample_to_fixed_length,
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib

from tqdm import tqdm
import pdb

from parameters.parameters import PreprocessParams


def cycle_preprocess(preprocess_params: PreprocessParams = None):

    # params load
    scaler_type = preprocess_params.scaler_type
    target_length = preprocess_params.target_length
    exclude_batteries = preprocess_params.exclude_batteries
    # 입력 경로
    input_folder = preprocess_params.original_data_path
    # resampled - 이제 안씀
    resampled_csv_folder = preprocess_params.resampled_csv_folder
    # 이상치 제거 결과 csv 경로
    outlier_output_folder = preprocess_params.outlier_cut_csv_path
    # 전처리 다 된 csv 경로
    preprocessed_csv_path = preprocess_params.preprocessed_csv_path
    # 전처리 다 된 .pt 경로
    preprocessed_data_path = preprocess_params.preprocessed_data_path
    outlier_threshold = preprocess_params.outlier_threshold

    # P1
    df_cleaned, total_df = process_and_save_outlier_data(
        exclude_batteries,
        input_folder,
        outlier_output_folder,
        outlier_threshold=outlier_threshold,
    )

    # P2 variables
    val_ratio = 0.2
    test_ratio = 0.2

    # P2 : prepare_dataset.py
    # 1. P1의 통합된 데이터프레임을 사용하여 피쳐 별 스케일 정규화를 진행
    # 이 때 file_index 컬럼은 스케일링에서 제외 후 다시 추가
    file_index_col = df_cleaned["file_index"].values
    df_cleaned = df_cleaned.drop(columns="file_index")

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

    # 다시 scaled_df에 저장했던 file_index 컬럼 복원
    scaled_df["file_index"] = file_index_col

    # 2. 정규화를 마친 통합 데이터프레임을 파일 단위로 분리 후
    grouped_dfs = grouping_df(scaled_df)
    resampled_dfs = {}

    # 3. 각 파일을 256개 샘플로 리샘플링하고 resampled_dfs에 저장
    for file_index, discharge_df in tqdm(grouped_dfs.items()):
        resampled_df = resample_to_fixed_length(
            discharge_df,
            target_length=target_length,
            # resampled_output_folder=resampled_csv_folder,
        )
        resampled_dfs[file_index] = resampled_df

    # 3.1 resampled_dfs 순회하면서 전처리 완료된 사이클 별 csv 파일 저장
    # total_preprocessed_csv_folder = f"cycle_preprocess/csv/total_preprocessed/processed_{scaler_type}"
    os.makedirs(preprocessed_csv_path, exist_ok=True)
    for file_index, resampled_df in resampled_dfs.items():
        output_path = os.path.join(preprocessed_csv_path, f"{int(file_index):05d}.csv")
        resampled_df.to_csv(output_path, index=False)
        # print(f"파일 저장 완료: {output_path} ({len(resampled_df)} rows)")

    # 4. 각 샘플들을 텐서로 변환
    # resmapled_dfs를 모두 합쳐서 하나의 데이터프레임으로 변경
    resampled_total_df = pd.concat(resampled_dfs.values(), ignore_index=True)

    # 4. 데이터를 train/val/test로 분할 (6:2:2)
    train_data, val_data, test_data, train_indices, val_indices, test_indices = (
        split_and_transform_data(
            resampled_total_df,
            list(grouped_dfs.keys()),
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
    )

    # 5. 학습/검증/테스트 데이터를 데이터셋으로 바꾸고,전처리에 사용한 스케일러 객체를 pickle, pt로 저장
    os.makedirs(preprocessed_data_path, exist_ok=True)

    # 파일 인덱스 정보도 함께 저장
    indices_info = {
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
    }
    with open(os.path.join(preprocessed_data_path, "file_indices.pkl"), "wb") as f:
        pickle.dump(indices_info, f)

    save_tensor_dataset(train_data, val_data, test_data, scaler, preprocessed_data_path)


if __name__ == "__main__":
    print("사이클 전처리 시작")
    cycle_preprocess()
    print("사이클 전처리 완료")
