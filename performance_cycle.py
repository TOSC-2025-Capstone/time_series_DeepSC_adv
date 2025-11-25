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
     * RMSE (Mean Absolute Percentage Error)
   - 성능 리포트 저장 (save_performance_report)

주요 기능:
- 텐서 -> DataFrame 변환 및 스케일 복원
- 연속 데이터의 사이클 단위 분할
- 사이클별 성능 분석 및 시각화
- 상세한 성능 지표 계산 및 리포트 생성
"""

import os
import pdb
from typing import Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from cycle_preprocess.reverse_cycle_reshape import *
from models.transceiver import DeepSC

from torch.utils.data import DataLoader, TensorDataset

from parameters.model_parameters import model_params
# 기타 매개변수, 모델 파라미터 모두 가져오기
from parameters.model_parameters import *
from parameters.parameters import ReconstructParams, TestParams
import parameters.parameters as p

from tqdm import tqdm

from train import normalize_snr_to_range

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

    result_df =  pd.DataFrame(data_original_scale, columns=feature_names).drop(columns=["cycle_sequence"])

    return result_df


def split_to_cycles(df_original, target_length=None, file_indices=None):
    """
    연속된 데이터프레임을 사이클 단위로 분할

    Args:
        df_original (pd.DataFrame): 연속된 데이터가 있는 데이터프레임
        target_length (int): 각 사이클의 길이 (기본값: 256)
        file_indices (list): 테스트 세트의 파일 인덱스 리스트

    Returns:
        dict: 파일 인덱스를 키로 하고 해당 사이클의 데이터프레임을 값으로 하는 딕셔너리
    """
    # 사이클의 개수 계산 (역정규화하느라 합쳐놓은 데이터 256row씩으로 다시 나누기)
    n_cycles = len(df_original) // target_length
    cycle_dfs = {}


    if file_indices is None or len(file_indices) != n_cycles:
        print(
            "경고: 유효한 file_indices가 제공되지 않았습니다. 순차적 인덱스를 사용합니다."
        )
        file_indices = list(range(1, n_cycles + 1))

    for i, file_idx in enumerate(file_indices):
        start_idx = i * target_length
        end_idx = (i + 1) * target_length
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
    plt.figure(figsize=(18, 10))
    for i, col in enumerate(feature_cols):
        plt.subplot(2, 4, i + 1)
        plt.plot(original_df[col], label="Original", alpha=0.7)
        plt.plot(reconstructed_df[col], label="Reconstructed", alpha=0.7)
        plt.title(col)
        plt.legend()
        plt.grid(True)
    plt.suptitle(f"Cycle Comparison: {base}")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(save_fig_dir, f"{base}_compare.png"), dpi=200)
    plt.close()


def calculate_performance_metrics(original_df, reconstructed_df, feature_cols):
    """
    원래 스케일로 복원 성능 지표 계산 (MSE, MAE, RMSE)

    Args:
        original_df (pd.DataFrame): 원본 사이클 데이터
        reconstructed_df (pd.DataFrame): 복원된 사이클 데이터
        feature_cols (list): 특성 컬럼 이름 리스트

    Returns:
        dict: 각 특성별 성능 지표
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


def post_process(tensor_data, scaler, preprocessed_folder, target_length, tensor_type):
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
    indices_path = os.path.join(preprocessed_folder, "file_indices.pkl")
    file_indices = None
    if os.path.exists(indices_path):
        with open(indices_path, "rb") as f:
            indices_info = pickle.load(f)
            file_indices = indices_info[f"{tensor_type}_indices"]
            # file_indices = indices_info["test_indices"]
    print("후처리 시작...")

    # 1. 텐서 데이터 역변환
    df_original = inverse_transform_tensor(tensor_data, scaler, preprocessed_folder)
    print(f"텐서 데이터 역변환 완료 (shape: {df_original.shape})")

    # 2. 사이클 단위로 분할
    cycle_dfs = split_to_cycles(
        df_original, target_length=target_length, file_indices=file_indices
    )
    print(f"총 {len(cycle_dfs)}개의 사이클로 분할 완료")
    print(f"파일 인덱스: {sorted(cycle_dfs.keys())}")

    return cycle_dfs


def total_performance_plot(feature_cols, all_metrics, save_dir):
    # 전체 성능 시각화
    plt.figure(figsize=(20, 15))
    metrics_names = ["MSE", "MAE", "RMSE"]

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
                            "Mean": [np.nanmean(values)],
                            "Std": [np.nanstd(values)],
                            "Min": [np.nanmin(values)],
                            "Max": [np.nanmax(values)],
                        }
                    ),
                ],
                ignore_index=True,
            )

    stats_df.to_csv(os.path.join(save_dir, "performance_statistics.csv"), index=False)

def evaluate_model_two_pass(model, tensor_data, device, test_snr_db):
    """
    Two-Pass 방식으로 모델 평가

    Pass 1: TX - 전력 정규화 계수 계산
    Pass 2: RX - 계수를 이용해 디코딩

    Args:
        model: 평가할 모델
        tensor_data: 입력 텐서 [batch, seq, features]
        device: cuda/cpu
        test_snr_db: 테스트 SNR (dB)

    Returns:
        all_output_tensors: 복원된 텐서 [total_samples, 512, 8]
    """
    from torch.cuda.amp import autocast
    from utils import power_normalize

    test_snr_normalized = normalize_snr_to_range(test_snr_db)
    batch_size = p.target_length // p.segment_length_n

    # SNR 레이블 추가
    batch_size_tensor, seq_len, features = tensor_data.shape
    main_features = tensor_data[:, :, :-1]  # [batch, seq, 8]
    file_index = tensor_data[:, :, -1:]     # [batch, seq, 1]

    snr_label = torch.full((batch_size_tensor, seq_len, 1), test_snr_normalized)
    tensor_with_snr = torch.cat([main_features, snr_label, file_index], dim=-1)

    # DataLoader 생성
    tensor_loader = DataLoader(
        tensor_with_snr,
        batch_size=batch_size,
        shuffle=False
    )

    # ========================================
    # PASS 1: TX - 전력 정규화 계수 계산
    # ========================================
    print(f"\n[Pass 1] Computing power normalization factor...")

    all_tx_signals = []
    all_input_batches = []

    model.eval()
    with torch.no_grad():
        pbar_pass1 = tqdm(tensor_loader, desc=f"TX Pass (SNR={test_snr_db}dB)")

        for batch in pbar_pass1:
            batch = batch.to(device)

            # file_index 제거: [batch, seq, 10] -> [batch, seq, 9]
            batch_without_file_idx = batch[:, :, :-1]
            data_8d = batch_without_file_idx[:, :, :-1]  # [batch, seq, 8]

            # 인코딩
            p.is_train_phase = False
            model.snr_db = test_snr_db

            if model.model_type == 'deepsc':
                encoded = model.encoder(batch_without_file_idx, None)
            else:
                encoded, _ = model.encoder(batch_without_file_idx)

            compressed = model.time_compressor(encoded)
            channel_encoded = model.channel_encoder(compressed)

            # CPU로 이동 (메모리 절약)
            all_tx_signals.append(channel_encoded.cpu())
            all_input_batches.append(data_8d.cpu())

            # GPU 메모리 해제
            del batch, batch_without_file_idx, data_8d, encoded, compressed, channel_encoded
            torch.cuda.empty_cache()

    # 전체 TX 신호 concat 및 정규화 계수 계산
    all_tx_concat = torch.cat(all_tx_signals, dim=0)  # CPU에서

    # 정규화 계수 계산 (학습과 동일)
    power = torch.mean(all_tx_concat ** 2)
    normalization_factor = torch.sqrt(power)

    pdb.set_trace()

    print(f"Computed normalization factor: {normalization_factor.item():.6f}")

    # 정규화된 TX 신호 생성
    total_tx_normalized = all_tx_concat / normalization_factor

    # 메모리 정리
    del all_tx_signals, all_tx_concat

    # ========================================
    # PASS 2: RX - 디코딩
    # ========================================
    print(f"[Pass 2] Decoding with normalized signals...")

    # 정규화된 TX 신호 DataLoader
    tx_loader = DataLoader(
        TensorDataset(total_tx_normalized),
        batch_size=batch_size,
        shuffle=False
    )

    # 입력 데이터 DataLoader
    all_data_8d = torch.cat(all_input_batches, dim=0)
    data_loader = DataLoader(
        TensorDataset(all_data_8d),
        batch_size=batch_size,
        shuffle=False
    )

    all_output_tensors = []

    with torch.no_grad():
        pbar_pass2 = tqdm(
            zip(tx_loader, data_loader),
            total=len(tx_loader),
            desc=f"RX Pass (SNR={test_snr_db}dB)"
        )

        for (tx_batch,), (data_batch,) in pbar_pass2:
            tx_batch = tx_batch.to(device)

            # 채널 통과
            with autocast():
                rx_sig = model.channels.Rayleigh(tx_batch, test_snr_db)

                # 디코딩
                channel_decoded = model.channel_decoder(rx_sig)
                decompressed = model.time_decompressor(channel_decoded)

                if model.model_type == 'deepsc':
                    output = model.decoder(decompressed, use_mask=True)
                else:
                    output, _ = model.decoder(decompressed)

                final_output = model.output_projection(output)
                final_output = final_output.permute(0, 2, 1)
                final_output = model.output_time_projection(final_output)
                final_output = final_output.permute(0, 2, 1)

                # SNR 레이블 제거: [batch, seq, 9] -> [batch, seq, 8]
                output_8d = final_output[:, :, :-1]

                # 배치 차원 펼치기
                output_8d = output_8d.contiguous().view(-1, 512, 8)

                all_output_tensors.append(output_8d.cpu())

            # GPU 메모리 해제
            del tx_batch, rx_sig, channel_decoded, decompressed
            del output, final_output, output_8d
            torch.cuda.empty_cache()

    # 전체 출력 결합
    final_output_tensor = torch.cat(all_output_tensors, dim=0)

    print(f"✓ Evaluation complete: {final_output_tensor.shape}")

    return final_output_tensor

def performance_cycle(
    params: Union[TestParams, ReconstructParams],
    model=None,
    device=None,
    is_full_reconstruct=False,
):
    """
    메인 평가 함수 (Two-Pass 방식 적용)
    """
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
    preprocessed_folder = params.preprocessed_path
    target_length = params.target_length
    scaler = joblib.load(scaler_path)

    feature_cols = params.feature_cols.copy()

    # 저장 경로 설정
    save_performance_dir = params.save_performance_dir
    save_reconstruction_dir = params.save_reconstruct_dir
    os.makedirs(save_performance_dir, exist_ok=True)
    os.makedirs(save_reconstruction_dir, exist_ok=True)

    # 2. 디바이스 및 모델 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model is None:
        print("모델을 전달해주세요!")
        return

    # 테스트용 SNR 설정
    test_snr_db = model_params.get("snr_db", 3)
    print(f"\n테스트 SNR 설정: {test_snr_db}dB (normalized: {normalize_snr_to_range(test_snr_db):.2f})")

    # 3. 각 데이터셋에 대해 평가 수행
    for idx, tensor_data in enumerate(tensor_list):
        print(f"\n{'='*60}")
        print(f"Evaluating {tensor_type_list[idx].upper()} dataset...")
        print(f"{'='*60}")

        # ===== Two-Pass 평가 =====
        all_output_tensors = evaluate_model_two_pass(
            model=model,
            tensor_data=tensor_data,
            device=device,
            test_snr_db=test_snr_db
        )

        # 복원된 사이클 얻기
        post_processed_cycles = post_process(
            tensor_data=all_output_tensors,
            scaler=scaler,
            preprocessed_folder=preprocessed_folder,
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
        for cycle_idx, reconstructed_df in post_processed_cycles.items():
            reconstruct_count += 1

            # 원본 데이터 로드
            original_path = os.path.join(
                params.csv_origin_path, f"{int(cycle_idx):05d}.csv"
            )

            if os.path.exists(original_path):
                original_df = pd.read_csv(original_path)
                original_df = original_df.drop(columns=["file_index"])
                original_df["battery_id"] = original_df["battery_id"].str.replace(r'\D', '', regex=True).astype(int)

                if reconstruct_count % 100 == 0:
                    print(f"사이클 {cycle_idx} 원본 데이터 로드 완료 (shape: {original_df.shape})")

                # 특성 이름은 reconstructed_df의 컬럼 순서 사용
                feature_cols = reconstructed_df.columns.tolist()

                # reverse sampling (256 -> 각 사이클 원래 길이)
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
                        cycle_idx,
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
                print(f"경고: 사이클 {cycle_idx}의 원본 데이터를 찾을 수 없습니다: {original_path}")

        # 전체 성능 플롯
        if is_full_reconstruct == False:
            total_performance_plot(feature_cols, all_metrics, save_performance_dir)

    print(f"\n{'='*60}")
    print("✓ All evaluations completed!")
    print(f"{'='*60}")

    # if is_full_reconstruct == False:
    #     total_performance_plot(feature_cols, all_metrics, save_performance_dir)


# if __name__ == "__main__":
# performance_cycle(model=model, device=device)
