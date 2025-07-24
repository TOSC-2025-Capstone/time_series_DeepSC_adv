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
process_discharge_files 함수:

metadata에서 discharge 파일 목록 추출
각 파일을 읽어서 256개 샘플로 리샘플링
결과를 preprocessed/resampled_256/ 폴더에 저장
"""


# P2.3
def resample_to_fixed_length(df, target_length=256):
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

    # # cycle_idx는 원래 값 유지
    # if "cycle_idx" in df.columns:
    #     resampled_data["cycle_idx"] = [df["cycle_idx"].iloc[0]] * target_length

    return pd.DataFrame(resampled_data)


def process_discharge_files(input_folder, output_folder):
    # 1. metadata에서 discharge 파일 정보 추출
    meta = pd.read_csv("original_dataset/metadata.csv")
    discharge_data = meta[meta["type"] == "discharge"]
    discharge_files = set(
        [f"{int(fname.split('.')[0]):05d}.csv" for fname in discharge_data["filename"]]
    )
    print(f"총 {len(discharge_files)}개의 discharge 파일 발견")

    # 2. 입력/출력 폴더 설정 (입력은 꼭 이상치 제거버전으로)
    # 이동

    # 3. 각 discharge 파일 처리
    for fname in sorted(discharge_files):
        fpath = os.path.join(input_folder, fname)
        if os.path.exists(fpath):
            print(f"처리 중: {fname}")
            # 파일 읽기
            df = pd.read_csv(fpath)

            # 256개 샘플로 리샘플링
            resampled_df = resample_to_fixed_length(df, target_length=256)

            # 결과 저장
            output_path = os.path.join(output_folder, fname)
            resampled_df.to_csv(output_path, index=False)
            print(f"완료: {fname} ({len(df)} -> 256 samples)")
        else:
            print(f"파일 없음: {fname}")


if __name__ == "__main__":
    input_folder = "cycle_preprocess/csv/outlier_cut/"
    output_folder = "cycle_preprocess/analysis/reshaped/resampled_256/"
    os.makedirs(output_folder, exist_ok=True)
    process_discharge_files(input_folder, output_folder)
