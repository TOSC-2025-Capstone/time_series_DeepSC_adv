import numpy as np
from scipy import interpolate
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import pdb


def detect_and_interpolate_outliers(df, threshold=3):
    """
    Z-score 기반으로 이상치(threshold 이상)를 탐지하고 선형 보간으로 대체
    """
    df_cleaned = df.copy()

    pdb.set_trace()  # 디버깅용
    for column in df.columns:
        if column != "cycle_idx":
            # Z-score 계산
            mean_val = df[column].mean()
            std_val = df[column].std()

            # std가 0이면 모든 값이 같다는 의미이므로 처리가 필요없음
            if std_val == 0:
                continue

            z_scores = np.abs((df[column] - mean_val) / std_val)

            # 이상치 위치 파악
            outliers = z_scores > threshold

            if outliers.any():
                # 이상치 위치의 인덱스
                outlier_indices = np.where(outliers)[0]

                # 선형 보간을 위한 유효한 데이터 포인트
                valid_indices = np.where(~outliers)[0]

                # 유효한 값이 없는 경우 (모든 값이 이상치인 경우)
                if len(valid_indices) == 0:
                    print(f"경고: {column} - 모든 값이 이상치로 판단됨. 원래 값 유지.")
                    continue

                # 유효한 값이 하나뿐인 경우
                if len(valid_indices) == 1:
                    print(
                        f"경고: {column} - 유효한 값이 하나뿐입니다. 해당 값으로 채움."
                    )
                    df_cleaned.loc[outlier_indices, column] = df[column].iloc[
                        valid_indices[0]
                    ]
                    continue

                valid_values = df[column].iloc[valid_indices]

                try:
                    # 선형 보간 함수 생성
                    f = interpolate.interp1d(
                        valid_indices,
                        valid_values,
                        bounds_error=False,  # 범위를 벗어나는 외삽 허용
                        fill_value=(valid_values.iloc[0], valid_values.iloc[-1]),
                    )  # 끝점 처리

                    # 이상치를 보간된 값으로 대체
                    df_cleaned.loc[outlier_indices, column] = f(outlier_indices)
                except Exception as e:
                    print(
                        f"경고: {column} - 보간 중 오류 발생. 원래 값 유지. 오류: {str(e)}"
                    )
                    continue

    return df_cleaned


if __name__ == "__main__":
    print("이상치 제거 시작")
