"""
# performance_cycle.py (수정됨)

전체 프로세스 흐름:
1. [신규] 텐서 분할-추론-결합 (reconstruct_full_tensor)
   - (N, target_length, F) 텐서를 (N*k, n, F) 텐서로 분할
   - 분할된 텐서를 모델에 입력하여 (N*k, n, F) 출력 텐서 생성
   - 출력 텐서를 다시 결합(stitch)하여 (N, target_length, F) 복원 텐서 생성

2. 텐서 후처리 (post_process)
   - 모델 출력 텐서를 원본 스케일로 역변환
   - 연속된 데이터를 사이클 단위로 분할 (수정됨: 세그먼트 인덱스 -> 사이클 인덱스 정제)
   - 사이클별 CSV 파일 저장

3. 성능 평가 (Performance Evaluation)
   - (기존과 동일)
"""

import os
import pdb
from typing import Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import pickle
from tqdm import tqdm

from cycle_preprocess.reverse_cycle_reshape import *
from models.transceiver import DeepSC

# 기타 매개변수, 모델 파라미터 모두 가져오기
from parameters.model_parameters import *
from parameters.parameters import ReconstructParams, TestParams


def inverse_transform_tensor(tensor_data, scaler, preprocessed_folder):
    """
    모델 출력 텐서를 원래 스케일로 역변환
    [수정됨] CSV 경로 추적 로직 수정
    """

    # 1. 텐서를 2D 배열로 변환 (reshape)
    # .cpu()와 .detach()를 추가하여 그래프 분리 및 CPU 연산 보장
    data_2d = tensor_data.detach().cpu().reshape(-1, tensor_data.shape[-1]).numpy()

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
        print("경고: 스케일러에서 피쳐 이름을 찾을 수 없습니다. preprocessed_csv_path에서 로드합니다.")
        # [수정] preprocessed_folder (e.g., .../data/minmax/case_1)에서
        # 상위 폴더로 이동하여 CSV 경로를 찾음
        try:
            # preprocessed_folder (preprocessed_data_path)는 .../data/minmax/case_1 형태
            # 3번 상위로 이동: .../data/minmax -> .../data -> .../
            base_preprocessed_path = os.path.dirname(os.path.dirname(os.path.dirname(preprocessed_folder)))
            csv_path = os.path.join(base_preprocessed_path, "csv", "total_preprocessed")

            sample_file = os.listdir(csv_path)[0]
            feature_names = pd.read_csv(os.path.join(csv_path, sample_file)).columns
        except Exception as e:
            print(f"피쳐 이름 로드 실패: {e}. 경로: {preprocessed_folder}")
            # 임시방편
            num_features = data_original_scale.shape[1]
            feature_names = [f"feature_{i+1}" for i in range(num_features)]

    return pd.DataFrame(data_original_scale, columns=feature_names)


def split_to_cycles(df_original, target_length=None, file_indices=None):
    """
    연속된 데이터프레임을 사이클 단위로 분할
    """
    # 사이클의 개수 계산 (역정규화하느라 합쳐놓은 데이터 256row씩으로 다시 나누기)
    n_cycles = len(df_original) // target_length
    cycle_dfs = {}

    if file_indices is None or len(file_indices) != n_cycles:
        print(
            f"경고: file_indices 길이({len(file_indices) if file_indices else 0})와 "
            f"사이클 수({n_cycles})가 불일치합니다. 순차적 인덱스를 사용합니다."
        )
        file_indices = list(range(1, n_cycles + 1)) # [1, 2, 3, ...]

    for i, file_idx in enumerate(file_indices):
        start_idx = i * target_length
        end_idx = (i + 1) * target_length
        cycle_df = df_original.iloc[start_idx:end_idx].copy()
        cycle_dfs[file_idx] = cycle_df  # 실제 파일 인덱스(e.g., '00001')를 키로 사용

    return cycle_dfs


def visualize_cycle_performance(
    original_df, reconstructed_df, feature_cols, save_fig_dir, cycle_idx
):
    """
    원본 사이클과 복원된 사이클의 성능 비교 시각화 (기존 코드와 동일)
    """
    # cycle_idx가 문자열일 수 있으므로 int로 변환
    base = f"{int(cycle_idx):05d}"  # 5자리 숫자로 포맷팅

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
        original_range = original_df[col].max() - original_df[col].min()
        y_limit = original_range * 0.5 if original_range > 1e-6 else 1.0

        plt.plot(residual, label="Residual", color="orange", alpha=0.8)
        plt.title(f"Residual: {col}")
        plt.axhline(0, color="gray", linestyle="--", linewidth=1)
        if y_limit > 1e-6:
            plt.ylim(-y_limit, y_limit)  # y축 범위 설정
            plt.ylabel(f"Error (±{(y_limit/original_range*100):.1f}% of range)")
        plt.legend()
        plt.grid(True)

    plt.suptitle(f"Cycle Residuals: {base}")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(save_fig_dir, f"{base}_residual.png"), dpi=200)
    plt.close()

    # ... (기타 플롯 함수들은 기존과 동일하게 유지) ...
    # 3. 복원 오차율(%) 플롯 - 원본 데이터 범위 대비 상대 오차
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(feature_cols):
        plt.subplot(2, 3, i + 1)
        original_max = np.max(original_df[col])
        original_min = np.min(original_df[col])
        original_range = original_max - original_min
        diff = np.abs(original_df[col] - reconstructed_df[col])
        relative_error = None

        if col == "Current_measured" or col == "Current_load":
            y_true_binary = (original_df[col] >= 0.5).astype(int)
            y_pred_binary = (reconstructed_df[col] >= 0.5).astype(int)
            binary_diff = np.abs(y_true_binary - y_pred_binary)
            relative_error = binary_diff * 100
        else:
            # 0으로 나누기 방지
            safe_original = original_df[col].copy()
            safe_original[safe_original == 0] = 1e-9
            relative_error = (diff / safe_original) * 100

        plt.plot(relative_error, label="Relative Error", color="orange", alpha=0.8)
        plt.title(f"Relative Error: {col}")
        plt.ylabel("Error (%)")
        plt.ylim(0, 50)  # 0~50%로 제한
        plt.axhline(0, color="gray", linestyle="--", linewidth=1)
        plt.legend()
        plt.grid(True)
    plt.suptitle(f"Cycle Residual Percent: {base}")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(save_fig_dir, f"{base}_residual_percent.png"), dpi=200)
    plt.close()

    # 복원 오차율 (min-max scale로 구한 버전)
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(feature_cols):
        plt.subplot(2, 3, i + 1)
        original_max = np.max(original_df[col])
        original_min = np.min(original_df[col])
        original_range = original_max - original_min
        diff = np.abs(original_df[col] - reconstructed_df[col])
        relative_error = None

        if col == "Current_measured" or col == "Current_load":
            y_true_binary = (original_df[col] >= 0.5).astype(int)
            y_pred_binary = (reconstructed_df[col] >= 0.5).astype(int)
            binary_diff = np.abs(y_true_binary - y_pred_binary)
            relative_error = binary_diff * 100
        else:
            # 0으로 나누기 방지
            safe_original_range = original_range if original_range > 1e-9 else 1.0
            relative_error = (diff / safe_original_range) * 100

        plt.plot(relative_error, label="Relative Error", color="orange", alpha=0.8)
        plt.title(f"Relative Error: {col}")
        plt.ylabel("Error (% of data range)")
        plt.ylim(0, 50)  # 데이터 범위의 0~50%로 제한
        plt.axhline(0, color="gray", linestyle="--", linewidth=1)
        plt.legend()
        plt.grid(True)
    plt.suptitle(f"Cycle Residual Percent: {base}")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(
        os.path.join(save_fig_dir, f"{base}_residual_percent_minmax.png"), dpi=200
    )
    plt.close()

def calculate_performance_metrics(original_df, reconstructed_df, feature_cols):
    """
    원래 스케일로 복원 성능 지표 계산 (MSE, MAE, RMSE) (기존 코드와 동일)
    """
    metrics = {}
    epsilon = 1e-9  # 0으로 나누기 방지

    for col in feature_cols:
        true = original_df[col].values # 원본 df
        pred = reconstructed_df[col].values # 복원본 df

        mse = np.mean((true - pred) ** 2)
        mae = np.mean(np.abs(true - pred))
        RMSE = np.sqrt(np.mean((true - pred) ** 2))

        metrics[col] = {"MSE": mse, "MAE": mae, "RMSE": RMSE}

    return metrics


def save_performance_report(metrics, cycle_idx, save_dir):
    """
    성능 지표 리포트 저장 (기존 코드와 동일)
    """
    os.makedirs(save_dir, exist_ok=True)
    report_path = os.path.join(save_dir, f"{int(cycle_idx):05d}_performance.txt")

    with open(report_path, "w") as f:
        f.write(f"=== Cycle {int(cycle_idx):05d} Performance Report ===\n\n")
        for col, metric_values in metrics.items():
            f.write(f"\n[{col}]\n")
            for metric_name, value in metric_values.items():
                f.write(f"{metric_name}: {value:.4f}\n")


def post_process(tensor_data, scaler, preprocessed_folder, target_length, tensor_type):
    """
    모델 출력 텐서를 원본 데이터 형식으로 복원하는 메인 함수
    [수정됨] file_indices가 세그먼트 인덱스일 경우, 고유한 사이클 ID로 정제
    """
    # 파일 인덱스 정보 로드
    indices_path = os.path.join(preprocessed_folder, "file_indices.pkl")
    file_indices = None
    if os.path.exists(indices_path):
        with open(indices_path, "rb") as f:
            indices_info = pickle.load(f)
            file_indices = indices_info[f"{tensor_type}_indices"]

            # [신규] file_indices가 세그먼트 인덱스('00001_0') 형태일 경우
            if file_indices and '_' in str(file_indices[0]):
                print("세그먼트 인덱스를 고유한 원본 사이클 ID로 정제합니다.")
                # 고유한 ID를 순서대로 정렬하여 추출 (set을 사용하여 중복 제거)
                unique_ids = sorted(list(set([idx.split('_')[0] for idx in file_indices])))
                file_indices = unique_ids # file_indices = ['00001', '00002', ...]
                print(f"정제된 사이클 ID {len(file_indices)}개 로드 완료.")

    print("후처리 시작...")

    # 1. 텐서 데이터 역변환
    df_original = inverse_transform_tensor(tensor_data, scaler, preprocessed_folder)
    print(f"텐서 데이터 역변환 완료 (shape: {df_original.shape})")

    # 2. 사이클 단위로 분할
    cycle_dfs = split_to_cycles(
        df_original, target_length=target_length, file_indices=file_indices
    )
    print(f"총 {len(cycle_dfs)}개의 사이클로 분할 완료")
    if cycle_dfs:
        print(f"파일 인덱스 (Keys): {sorted(cycle_dfs.keys())[:10]}...") # 처음 10개만 표시
    else:
        print("경고: 분할된 사이클이 없습니다.")

    return cycle_dfs


def total_performance_plot(feature_cols, all_metrics, save_dir):
    """
    전체 성능 시각화 (기존 코드와 동일)
    """
    # 전체 성능 시각화
    plt.figure(figsize=(20, 15))
    metrics_names = ["MSE", "MAE", "RMSE"]

    for i, metric_name in enumerate(metrics_names):
        plt.subplot(3, 1, i + 1)

        for j, feature in enumerate(feature_cols):
            values = all_metrics[feature][metric_name]
            if not values: # 빈 리스트일 경우 스킵
                continue
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
            if not values: # 빈 리스트일 경우 스킵
                continue

            new_row = pd.DataFrame(
                {
                    "Feature": [feature],
                    "Metric": [metric],
                    "Mean": [np.nanmean(values)],
                    "Std": [np.nanstd(values)],
                    "Min": [np.nanmin(values)],
                    "Max": [np.nanmax(values)],
                }
            )
            stats_df = pd.concat([stats_df, new_row], ignore_index=True)

    stats_df.to_csv(os.path.join(save_dir, "performance_statistics.csv"), index=False)


# --- [신규] 분할-추론-결합(Split-Infer-Stitch) 헬퍼 함수 ---

def reconstruct_full_tensor(
    model, full_tensor_data, segment_length_n, batch_size, device
):
    """
    (N, target_length, F) 텐서를 (N*k, n, F) 세그먼트로 분할하고,
    모델 추론 후 다시 (N, target_length, F)로 결합합니다.

    Args:
        model (nn.Module): 추론에 사용할 모델
        full_tensor_data (torch.Tensor): (N_cycles, target_length, N_features)
        segment_length_n (int): 모델이 입력받는 시퀀스 길이 (n)
        batch_size (int): 추론 시 사용할 배치 크기
        device (torch.device): 디바이스

    Returns:
        torch.Tensor: 복원된 (N_cycles, target_length, N_features) 텐서
    """
    model.eval()

    # 1. 텐서 정보 추출
    num_cycles, target_length, num_features = full_tensor_data.shape
    n = segment_length_n

    # 사이클 당 세그먼트 수 계산
    num_segments_per_cycle = int(np.ceil(target_length / n))

    print(f"총 {num_cycles}개의 사이클을 {num_segments_per_cycle}개의 세그먼트(길이 {n})로 분할/추론/결합합니다.")

    # 2. 모든 사이클을 세그먼트로 분할
    all_segments = []
    for i in range(num_cycles):
        full_cycle = full_tensor_data[i]  # (target_length, F)

        for j in range(num_segments_per_cycle):
            start = j * n
            end = start + n
            segment = full_cycle[start:end]

            # 마지막 세그먼트 패딩
            if segment.shape[0] < n:
                padding_size = n - segment.shape[0]
                # (padding_size, num_features)
                padding = torch.zeros(padding_size, num_features, dtype=segment.dtype, device=segment.device)
                segment = torch.cat([segment, padding], dim=0)

            all_segments.append(segment)

    # (N_cycles * num_segments_per_cycle, n, N_features)
    # .to(device)는 DataLoader에서 처리하므로 여기서는 필요 없음
    segments_tensor = torch.stack(all_segments)

    # 3. DataLoader를 사용한 배치 추론
    segment_dataset = TensorDataset(segments_tensor)
    # shuffle=False가 중요 (순서 유지)f
    segment_loader = DataLoader(segment_dataset, batch_size=batch_size, shuffle=False)

    reconstructed_segments = []
    with torch.no_grad():
        pbar = tqdm(segment_loader, desc="Reconstructing segments")
        for batch in pbar:
            batch_segments = batch[0].to(device)
            output_segments = model(batch_segments)
            # pdb.set_trace()
            reconstructed_segments.append(output_segments.cpu())

    # (N_cycles * num_segments_per_cycle, n, N_features)
    all_reconstructed_segments = torch.cat(reconstructed_segments, dim=0)

    # 4. 세그먼트를 다시 전체 사이클로 결합 (Stitch)
    final_output_tensor = torch.zeros(num_cycles, target_length, num_features)

    segment_idx = 0
    for i in tqdm(range(num_cycles), desc="Stitching cycles"):
        stitched_cycle_segments = []
        for j in range(num_segments_per_cycle):
            stitched_cycle_segments.append(all_reconstructed_segments[segment_idx])
            segment_idx += 1

        # (num_segments_per_cycle * n, N_features)
        stitched_cycle_tensor = torch.cat(stitched_cycle_segments, dim=0)

        # 패딩 제거: 원본 target_length까지만 잘라냄
        final_output_tensor[i] = stitched_cycle_tensor[:target_length]

    print("전체 텐서 복원 완료.")
    return final_output_tensor


# --- [수정] 메인 performance_cycle 함수 ---

def performance_cycle(
    params: Union[TestParams, ReconstructParams],
    model=None,
    device=None,
    is_full_reconstruct=False,
):

    # 1. 데이터 및 메타 정보 로드
    train_tensor = None
    val_tensor = None
    test_tensor = None

    test_pt = params.test_pt
    test_data = torch.load(test_pt)
    test_tensor = test_data.tensors[0]

    # 전체 데이터 복원인 경우
    if is_full_reconstruct == True:
        train_pt = params.train_pt
        train_data = torch.load(train_pt)
        train_tensor = train_data.tensors[0]
        val_pt = params.val_pt
        val_data = torch.load(val_pt)
        val_tensor = val_data.tensors[0]

    tensor_list = None
    tensor_type_list = None
    if is_full_reconstruct:
        tensor_list = [train_tensor, val_tensor, test_tensor]
        tensor_type_list = ["train", "val", "test"]
    else:
        tensor_list = [test_tensor]
        tensor_type_list = ["test"]

    scaler_path = params.scaler_path
    preprocessed_folder = params.preprocessed_path # e.g., .../data/minmax/case_1
    target_length = params.target_length
    scaler = joblib.load(scaler_path)

    feature_cols = params.feature_cols.copy()

    # [추가] segment_length_n (n) 및 배치 크기 로드
    try:
        segment_length_n = params.segment_length_n
    except AttributeError:
        print(f"경고: {type(params).__name__}에 'segment_length_n'이 없습니다. 기본값 8을 사용합니다.")
        segment_length_n = 8 # n의 기본값 (train.py와 일치시킴)

    # 추론 시 사용할 배치 크기 (params에 있으면 사용, 없으면 64)
    batch_size = getattr(params, 'batch_size', 64)

    # 저장 경로 설정
    save_performance_dir = params.save_performance_dir
    save_reconstruction_dir = params.save_reconstruct_dir
    os.makedirs(save_performance_dir, exist_ok=True)
    os.makedirs(save_reconstruction_dir, exist_ok=True)

    # 2. 입력 형태 정의 및 모델 로드
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = test_tensor.shape[2]

    # 3. 전체 배터리 시계열 복원 및 성능 평가
    post_processed_cycles = None

    for idx, tensor_data in enumerate(tensor_list):
        if tensor_data is None:
            continue

        if model is None:
            print("모델을 전달해주세요!")
            return
        else:
            # [수정] 단순 모델 호출 대신 'reconstruct_full_tensor' 함수 사용
            print(f"'{tensor_type_list[idx]}' 데이터셋 전체 복원을 시작합니다...")
            output_tensor = reconstruct_full_tensor(
                model=model,
                full_tensor_data=tensor_data, # (N, target_length, F)
                segment_length_n=segment_length_n,
                batch_size=batch_size,
                device=device
            )
            # output_tensor shape: (N, target_length, F)

        # 복원된 사이클 얻기
        # [수정] post_process가 file_indices.pkl을 올바르게 찾도록 preprocessed_folder 경로 수정
        # params.preprocessed_path (e.g., .../data/minmax/case_1)
        # file_indices.pkl은 이 폴더(preprocessed_folder) 안에 있습니다.
        post_processed_cycles = post_process(
            tensor_data=output_tensor.cpu(),
            scaler=scaler,
            preprocessed_folder=preprocessed_folder, # .pkl 파일이 있는 경로
            target_length=target_length,
            tensor_type=tensor_type_list[idx],
        )

        print(f"사이클 복원 완료, 총 {len(post_processed_cycles)}개의 사이클")

        # 모든 사이클의 성능 지표를 저장할 딕셔너리
        all_metrics = {
            feature: {"MSE": [], "MAE": [], "RMSE": []} for feature in feature_cols
        }

        reconstruct_count = 0

        # 원본 데이터 로드 및 성능 평가
        # post_process가 file_indices를 정제했으므로 cycle_idx는 '00001' 등
        pbar_cycles = tqdm(post_processed_cycles.items(), desc="Evaluating cycles")
        for cycle_idx, reconstructed_df in pbar_cycles:
            reconstruct_count += 1

            pbar_cycles.set_description(f"Evaluating cycle {cycle_idx}")

            # 원본 데이터 로드 (길이 제각각)
            original_path = os.path.join(
                params.csv_origin_path, f"{int(cycle_idx):05d}.csv"
            )
            if os.path.exists(original_path):
                original_df = pd.read_csv(original_path)

                # if reconstruct_count % 100 == 0: # tqdm 사용으로 대체
                #     print(
                #         f"사이클 {cycle_idx} 원본 데이터 로드 완료 (shape: {original_df.shape})"
                #     )

                # 특성 이름은 reconstructed_df의 컬럼 순서 사용
                feature_cols = reconstructed_df.columns.tolist()

                # reverse sampling (target_length -> 각 사이클 원래 길이)
                reversed_df = reverse_resample(reconstructed_df, len(original_df))

                # 사이클 데이터프레임을 CSV로 저장
                reversed_df.to_csv(
                    os.path.join(
                        save_reconstruction_dir,
                        f"{int(cycle_idx):05d}_reconstructed.csv",
                    ),
                    index=False,
                )

                # 시각화 (100개당 하나)
                if is_full_reconstruct == False and reconstruct_count % 100 == 0:
                    visualize_cycle_performance(
                        original_df,
                        reversed_df,
                        feature_cols,
                        save_performance_dir,
                        cycle_idx, # cycle_idx는 '00001' 같은 문자열일 수 있음
                    )

                # 성능 지표 계산 및 저장
                metrics = calculate_performance_metrics(
                    original_df, reversed_df, feature_cols
                )

                # 각 feature의 metrics를 저장
                for feature in feature_cols:
                    for metric_name in ["MSE", "MAE", "RMSE"]:
                        all_metrics[feature][metric_name].append(
                            metrics[feature][metric_name]
                        )

            else:
                print(
                    f"경고: 사이클 {cycle_idx}의 원본 데이터를 찾을 수 없습니다: {original_path}"
                )

    if is_full_reconstruct == False:
        total_performance_plot(feature_cols, all_metrics, save_performance_dir)


# if __name__ == "__main__":
#     # 예:
#     # my_params = TestParams()
#     # my_params.segment_length_n = 8 # train.py와 동일한 n 값 설정
#     # model = ... (모델 로드)
#     # performance_cycle(params=my_params, model=model, device=device)
#     pass
