"""
# performance_cycle.py

전체 프로세스 흐름:
1. 텐서 후처리 (post_process)
   - 모델 출력 텐서를 원본 스케일로 역변환
   - 연속된 데이터를 사이클 단위로 분할
   - 사이클별 CSV 파일 저장

2. 성능 평가 (Performance Evaluation)
   - 시각화 (visualize_cycle_performance)
     * 원본-복원 비교 그래프
     * Residual(오차) 시계열 그래프
     * 복원 오차율(%) 그래프
   - 성능 지표 계산 (calculate_performance_metrics)
     * MSE (Mean Squared Error)
     * MAE (Mean Absolute Error)
     * MAPE (Mean Absolute Percentage Error)
   - 성능 리포트 저장 (save_performance_report)

주요 기능:
- 텐서 -> DataFrame 변환 및 스케일 복원
- 연속 데이터의 사이클 단위 분할
- 사이클별 성능 분석 및 시각화
- 상세한 성능 지표 계산 및 리포트 생성
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import joblib
import pdb
from models.transceiver import DeepSC

# 기타 매개변수, 모델 파라미터 모두 가져오기
from parameters.model_parameters import *
from cycle_preprocess.reverse_cycle_reshape import *

from parameters.parameters import reconstructed_data_path


def inverse_transform_tensor(tensor_data, scaler, preprocessed_folder):
    """
    모델 출력 텐서를 원래 스케일로 역변환

    Args:
        tensor_data (torch.Tensor): 모델이 출력한 텐서 데이터 (batch_size, sequence_length, n_features)
        scaler: 전처리에 사용된 스케일러 객체
        preprocessed_folder: 전처리된 데이터가 저장된 폴더 경로

    Returns:
        pd.DataFrame: 역변환된 데이터프레임
    """

    # 1. 텐서를 2D 배열로 변환 (reshape)
    data_2d = tensor_data.reshape(-1, tensor_data.shape[-1]).numpy()

    # 2. 스케일러에서 특성 이름 가져오기
    feature_names = (
        list(scaler.feature_names_in_)
        if hasattr(scaler, "feature_names_in_")
        else (
            list(scaler.get_feature_names_out())
            if hasattr(scaler, "get_feature_names_out")
            else None
        )
    )

    # 3. 스케일러로 역변환
    data_original_scale = scaler.inverse_transform(data_2d)

    # 4. DataFrame으로 변환 (스케일러의 특성 순서 사용)
    if feature_names is None:
        # 스케일러에서 특성 이름을 가져올 수 없는 경우, 샘플 파일에서 가져오기
        sample_file = os.listdir(
            os.path.join(preprocessed_folder, "csv/total_preprocessed")
        )[0]
        feature_names = pd.read_csv(
            os.path.join(preprocessed_folder, "csv/total_preprocessed", sample_file)
        ).columns

    return pd.DataFrame(data_original_scale, columns=feature_names)


def split_to_cycles(df_original, sequence_length=256, file_indices=None):
    """
    연속된 데이터프레임을 사이클 단위로 분할

    Args:
        df_original (pd.DataFrame): 연속된 데이터가 있는 데이터프레임
        sequence_length (int): 각 사이클의 길이 (기본값: 256)
        file_indices (list): 테스트 세트의 파일 인덱스 리스트

    Returns:
        dict: 파일 인덱스를 키로 하고 해당 사이클의 데이터프레임을 값으로 하는 딕셔너리
    """
    # 사이클의 개수 계산 (역정규화하느라 합쳐놓은 데이터 256row씩으로 다시 나누기)
    n_cycles = len(df_original) // sequence_length
    cycle_dfs = {}

    if file_indices is None or len(file_indices) != n_cycles:
        print(
            "경고: 유효한 file_indices가 제공되지 않았습니다. 순차적 인덱스를 사용합니다."
        )
        file_indices = list(range(1, n_cycles + 1))

    for i, file_idx in enumerate(file_indices):
        start_idx = i * sequence_length
        end_idx = (i + 1) * sequence_length
        cycle_df = df_original.iloc[start_idx:end_idx].copy()
        cycle_dfs[file_idx] = cycle_df  # 실제 파일 인덱스를 키로 사용

    return cycle_dfs


def visualize_cycle_performance(
    original_df, reconstructed_df, feature_cols, save_fig_dir, cycle_idx
):
    """
    원본 사이클과 복원된 사이클의 성능 비교 시각화

    Args:
        original_df (pd.DataFrame): 원본 사이클 데이터
        reconstructed_df (pd.DataFrame): 복원된 사이클 데이터
        feature_cols (list): 특성 컬럼 이름 리스트
        save_fig_dir (str): 그래프 저장 경로
        cycle_idx (int): 현재 사이클 인덱스
    """
    base = f"{cycle_idx:05d}"  # 5자리 숫자로 포맷팅

    # 각 파일별로 폴더 생성
    save_fig_dir = os.path.join(save_fig_dir, base)
    os.makedirs(save_fig_dir, exist_ok=True)

    # 1. 원본-복원 비교 플롯
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(feature_cols):
        plt.subplot(2, 3, i + 1)
        plt.plot(original_df[col], label="Original", alpha=0.7)
        plt.plot(reconstructed_df[col], label="Reconstructed", alpha=0.7)
        plt.title(col)
        plt.legend()
        plt.grid(True)
    plt.suptitle(f"Cycle Comparison: {base}")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(save_fig_dir, f"{base}_compare.png"), dpi=200)
    plt.close()

    # 2. Residual(오차) 시계열 플롯
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(feature_cols):
        plt.subplot(2, 3, i + 1)
        residual = original_df[col] - reconstructed_df[col]

        # 원본 데이터의 범위 계산
        original_range = original_df[col].max() - original_df[col].min()
        y_limit = original_range * 0.5  # 원본 데이터 범위의 ±50%로 설정

        plt.plot(residual, label="Residual", color="orange", alpha=0.8)
        plt.title(f"Residual: {col}")
        plt.axhline(0, color="gray", linestyle="--", linewidth=1)
        plt.ylim(-y_limit, y_limit)  # y축 범위 설정
        plt.legend()
        plt.grid(True)

        # y축에 원본 데이터 범위의 백분율 표시
        plt.ylabel(f"Error (±{(y_limit/original_range*100):.1f}% of range)")

    plt.suptitle(f"Cycle Residuals: {base}")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(save_fig_dir, f"{base}_residual.png"), dpi=200)
    plt.close()

    # 3. 복원 오차율(%) 플롯 - 원본 데이터 범위 대비 상대 오차
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(feature_cols):
        plt.subplot(2, 3, i + 1)

        # 원본 데이터의 범위 계산
        original_max = np.max(original_df[col])
        original_min = np.min(original_df[col])
        original_range = original_max - original_min

        # 절대 오차 계산
        diff = np.abs(original_df[col] - reconstructed_df[col])

        # 오차를 원본 데이터 범위에 대한 비율로 표시 (백분율)
        relative_error = (diff / original_range) * 100

        plt.plot(relative_error, label="Relative Error", color="orange", alpha=0.8)
        plt.title(f"Relative Error: {col}")
        plt.ylabel("Error (% of data range)")
        plt.ylim(0, 50)  # 데이터 범위의 0~50%로 제한
        plt.axhline(0, color="gray", linestyle="--", linewidth=1)
        plt.legend()
        plt.grid(True)
    plt.suptitle(f"Cycle Residual Percent: {base}")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(save_fig_dir, f"{base}_residual_percent.png"), dpi=200)
    plt.close()


def calculate_performance_metrics(original_df, reconstructed_df, feature_cols):
    """
    원래 스케일로 복원 성능 지표 계산 (MSE, MAE, MAPE)

    Args:
        original_df (pd.DataFrame): 원본 사이클 데이터
        reconstructed_df (pd.DataFrame): 복원된 사이클 데이터
        feature_cols (list): 특성 컬럼 이름 리스트

    Returns:
        dict: 각 특성별 성능 지표
    """
    metrics = {}
    epsilon = 1e-9  # 0으로 나누기 방지

    # pdb.set_trace()

    for col in feature_cols:
        true = original_df[col].values
        pred = reconstructed_df[col].values

        mse = np.mean((true - pred) ** 2)
        mae = np.mean(np.abs(true - pred))
        mape = np.mean(np.abs((true - pred) / (np.abs(true) + epsilon))) * 100

        metrics[col] = {"MSE": mse, "MAE": mae, "MAPE": mape}

    return metrics


def save_performance_report(metrics, cycle_idx, save_dir):
    """
    성능 지표 리포트 저장

    Args:
        metrics (dict): 계산된 성능 지표
        cycle_idx (int): 현재 사이클 인덱스
        save_dir (str): 저장 경로
    """
    os.makedirs(save_dir, exist_ok=True)
    report_path = os.path.join(save_dir, f"{cycle_idx:05d}_performance.txt")

    with open(report_path, "w") as f:
        f.write(f"=== Cycle {cycle_idx:05d} Performance Report ===\n\n")
        for col, metric_values in metrics.items():
            f.write(f"\n[{col}]\n")
            for metric_name, value in metric_values.items():
                f.write(f"{metric_name}: {value:.4f}\n")


def post_process(tensor_data, scaler, preprocessed_folder="cycle_preprocess/"):
    """
    모델 출력 텐서를 원본 데이터 형식으로 복원하는 메인 함수

    Args:
        tensor_data (torch.Tensor): 모델 출력 텐서
        scaler: 전처리에 사용된 스케일러 객체
        preprocessed_folder (str): 전처리 데이터가 저장된 폴더 경로

    Returns:
        dict: 사이클별로 복원된 데이터프레임 딕셔너리
    """
    # 파일 인덱스 정보 로드
    indices_path = os.path.join(
        preprocessed_folder, "total_preprocessed/file_indices.pkl"
    )
    if os.path.exists(indices_path):
        with open(indices_path, "rb") as f:
            indices_info = pickle.load(f)
            test_indices = indices_info["test_indices"]
    print("후처리 시작...")

    # 1. 텐서 데이터 역변환
    df_original = inverse_transform_tensor(tensor_data, scaler, preprocessed_folder)
    print(f"텐서 데이터 역변환 완료 (shape: {df_original.shape})")

    # 2. 사이클 단위로 분할
    cycle_dfs = split_to_cycles(df_original, file_indices=test_indices)
    print(f"총 {len(cycle_dfs)}개의 사이클로 분할 완료")
    print(f"파일 인덱스: {sorted(cycle_dfs.keys())}")

    return cycle_dfs


def total_performance_plot(feature_cols, all_metrics, save_dir):
    # 전체 성능 시각화
    plt.figure(figsize=(20, 15))
    metrics_names = ["MSE", "MAE", "MAPE"]

    for i, metric_name in enumerate(metrics_names):
        plt.subplot(3, 1, i + 1)

        for j, feature in enumerate(feature_cols):
            values = all_metrics[feature][metric_name]
            x = np.ones_like(values) * j + np.random.normal(
                0, 0.1, len(values)
            )  # 산점도 점들을 약간 흩뿌림

            # 산점도
            plt.scatter(x, values, alpha=0.3, label=f"{feature}")

            # 평균선
            mean_value = np.mean(values)
            plt.hlines(mean_value, j - 0.3, j + 0.3, colors="red", linestyles="solid")

        plt.title(f"{metric_name} Distribution Across Features")
        plt.grid(True, alpha=0.3)
        plt.xticks(range(len(feature_cols)), feature_cols, rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "all_metrics_distribution.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # 성능 통계 저장
    stats_df = pd.DataFrame(columns=["Feature", "Metric", "Mean", "Std", "Min", "Max"])
    for feature in feature_cols:
        for metric in metrics_names:
            values = all_metrics[feature][metric]
            stats_df = pd.concat(
                [
                    stats_df,
                    pd.DataFrame(
                        {
                            "Feature": [feature],
                            "Metric": [metric],
                            "Mean": [np.mean(values)],
                            "Std": [np.std(values)],
                            "Min": [np.min(values)],
                            "Max": [np.max(values)],
                        }
                    ),
                ],
                ignore_index=True,
            )

    stats_df.to_csv(os.path.join(save_dir, "performance_statistics.csv"), index=False)


def performance_cycle(model=None, device=None):
    train_pt = "cycle_preprocess/total_preprocessed/processed_minmax/train_data.pt"
    test_pt = "cycle_preprocess/total_preprocessed/processed_minmax/test_data.pt"
    scaler_path = "cycle_preprocess/total_preprocessed/processed_minmax/scaler.pkl"
    # model_checkpoint_path = "checkpoints/case_7.1/MSE/DeepSC_battery_epoch"

    # 1. 데이터 및 메타 정보 로드
    train_data = torch.load(train_pt)
    test_data = torch.load(test_pt)
    train_tensor = train_data.tensors[0]
    test_tensor = test_data.tensors[0]
    scaler = joblib.load(scaler_path)
    train_len = len(train_data.tensors[0])

    # 스케일러가 학습된 순서대로 feature_cols 설정
    feature_cols = [
        "Voltage_measured",
        "Current_measured",
        "Temperature_measured",
        "Current_load",
        "Voltage_load",
        "Time",
    ]
    # 스케일러에서 학습된 특성 순서 가져오기
    if hasattr(scaler, "feature_names_in_"):
        feature_cols = list(scaler.feature_names_in_)
    elif hasattr(scaler, "get_feature_names_out"):
        feature_cols = list(scaler.get_feature_names_out())
    else:
        print("스케일러에서 특성 이름을 가져올 수 없습니다. 기본값 사용")

    # 2. 입력 형태 정의 및 모델 로드
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = test_tensor.shape[2]
    window_size = test_tensor.shape[1]

    save_dir = "cycle_preprocess/performance_test/"
    # cycle_idx = 1

    # 3. 전체 배터리 시계열 복원 및 성능 평가
    save_dir = "cycle_preprocess/performance_test/"
    os.makedirs(save_dir, exist_ok=True)

    # 복원된 사이클 얻기
    post_processed_cycles = post_process(
        tensor_data=test_tensor, scaler=scaler, preprocessed_folder="cycle_preprocess/"
    )

    print(f"사이클 복원 완료, 총 {len(post_processed_cycles)}개의 사이클")
    # 복원된 사이클 저장 경로 생성
    os.makedirs(reconstructed_data_path, exist_ok=True)
    # 복원된 사이클 csv로 저장
    for cycle_idx, cycle_df in post_processed_cycles.items():
        # 사이클 데이터프레임을 CSV로 저장
        cycle_df.to_csv(
            os.path.join(
                reconstructed_data_path, f"{int(cycle_idx):05d}_reconstructed.csv"
            ),
            index=False,
        )
        print(f"사이클 {cycle_idx} 복원 완료 및 저장")

    # 모든 사이클의 성능 지표를 저장할 딕셔너리
    all_metrics = {
        feature: {"MSE": [], "MAE": [], "MAPE": []} for feature in feature_cols
    }

    reconstruct_count = 0

    # 원본 데이터 로드 및 성능 평가
    for cycle_idx, reconstructed_df in post_processed_cycles.items():
        reconstruct_count += 1
        # 원본 데이터 로드 (길이 제각각)
        original_path = os.path.join(
            "cycle_preprocess/csv/outlier_cut/", f"{int(cycle_idx):05d}.csv"
        )
        if os.path.exists(original_path):
            original_df = pd.read_csv(original_path)

            if reconstruct_count % 100 == 0:
                print(
                    f"사이클 {cycle_idx} 원본 데이터 로드 완료 (shape: {original_df.shape})"
                )

            # 특성 이름은 reconstructed_df의 컬럼 순서 사용
            feature_cols = reconstructed_df.columns.tolist()

            # reverse sampling (256 -> 각 사이클 원래 길이)
            reversed_df = reverse_resample(reconstructed_df, len(original_df))

            # 시각화 (100개당 하나)
            if reconstruct_count % 100 == 0:
                visualize_cycle_performance(
                    original_df, reversed_df, feature_cols, save_dir, cycle_idx
                )

            # 성능 지표 계산 및 저장
            metrics = calculate_performance_metrics(
                original_df, reversed_df, feature_cols
            )

            # 각 feature의 metrics를 저장
            for feature in feature_cols:
                for metric_name in ["MSE", "MAE", "MAPE"]:
                    all_metrics[feature][metric_name].append(
                        metrics[feature][metric_name]
                    )

        else:
            print(
                f"경고: 사이클 {cycle_idx}의 원본 데이터를 찾을 수 없습니다: {original_path}"
            )

    total_performance_plot(feature_cols, all_metrics, save_dir)


# if __name__ == "__main__":
# performance_cycle(model=model, device=device)
