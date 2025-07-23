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


def calculate_error_metrics(original_df, restored_df):
    """
    원본 데이터와 복원된 데이터 간의 오차 계산
    """
    errors = {}
    for column in original_df.columns:
        if column != "cycle_idx":
            # Mean Absolute Error (MAE)
            mae = np.mean(
                np.abs(original_df[column].values - restored_df[column].values)
            )
            # Mean Squared Error (MSE)
            mse = np.mean(
                (original_df[column].values - restored_df[column].values) ** 2
            )
            # Root Mean Squared Error (RMSE)
            rmse = np.sqrt(mse)
            # Mean Absolute Percentage Error (MAPE)
            mape = (
                np.mean(
                    np.abs(
                        (original_df[column].values - restored_df[column].values)
                        / original_df[column].values
                    )
                )
                * 100
            )

            errors[column] = {"MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE": mape}
    return errors


def plot_comparison(original_df, restored_df, errors, fname, output_folder):
    """
    원본 데이터와 복원된 데이터의 비교 그래프 생성
    """
    columns = [col for col in original_df.columns if col != "cycle_idx"]
    n_cols = len(columns)

    plt.figure(figsize=(15, 5 * n_cols))

    for idx, column in enumerate(columns, 1):
        plt.subplot(n_cols, 1, idx)

        # 원본 데이터
        plt.plot(original_df[column].values, label="Original", alpha=0.7)
        # 복원된 데이터
        plt.plot(restored_df[column].values, label="Restored", alpha=0.7)

        plt.title(f'{column} Comparison\nMAPE: {errors[column]["MAPE"]:.2f}%')
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{fname}_comparison.png"))
    plt.close()


def process_files():
    # 1. metadata에서 discharge 파일 정보 추출
    meta = pd.read_csv("original_dataset/metadata.csv")
    discharge_data = meta[meta["type"] == "discharge"]
    discharge_files = set(
        [f"{int(fname.split('.')[0]):05d}.csv" for fname in discharge_data["filename"]]
    )
    print(f"총 {len(discharge_files)}개의 discharge 파일 발견")

    # 2. 폴더 경로 설정
    original_folder = "original_dataset/data/"
    compressed_folder = "cycle_preprocess/preprocessed/resampled_256/"
    output_folder = "cycle_preprocess/analysis/restoration_results/"
    os.makedirs(output_folder, exist_ok=True)

    # 결과 저장을 위한 데이터프레임
    all_errors = []

    # 3. 각 discharge 파일 처리
    for fname in sorted(discharge_files):
        original_path = os.path.join(original_folder, fname)
        compressed_path = os.path.join(compressed_folder, fname)

        if os.path.exists(original_path) and os.path.exists(compressed_path):
            print(f"처리 중: {fname}")

            # 원본 파일과 압축 파일 읽기
            original_df = pd.read_csv(original_path)
            compressed_df = pd.read_csv(compressed_path)

            # 압축 파일을 원래 길이로 복원
            restored_df = reverse_resample(compressed_df, len(original_df))

            # 오차 계산
            errors = calculate_error_metrics(original_df, restored_df)

            # 결과를 all_errors에 추가
            for column, metrics in errors.items():
                all_errors.append({"filename": fname, "column": column, **metrics})

            # 100개당 하나씩 비교 그래프 생성
            file_number = int(fname.split(".")[0])
            if file_number % 100 == 0:  # 파일 번호가 100의 배수일 때만 그래프 생성
                plot_comparison(original_df, restored_df, errors, fname, output_folder)
                print(f"비교 그래프 생성: {fname}")

            print(f"완료: {fname}")
        else:
            print(f"파일 없음: {fname}")

    # 4. 전체 오차 통계 저장
    errors_df = pd.DataFrame(all_errors)
    errors_df.to_csv(os.path.join(output_folder, "restoration_errors.csv"), index=False)

    # 5. 전체 오차 분포 시각화
    plt.figure(figsize=(15, 10))

    for idx, metric in enumerate(["MAE", "RMSE", "MAPE"], 1):
        plt.subplot(2, 2, idx)
        for column in errors_df["column"].unique():
            column_data = errors_df[errors_df["column"] == column][metric]
            plt.hist(column_data, bins=30, alpha=0.5, label=column)
        plt.title(f"{metric} Distribution")
        plt.xlabel(metric)
        plt.ylabel("Count")
        plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "error_distributions.png"))
    plt.close()

    print("\n=== 복원 완료 ===")
    print(f"결과가 {output_folder} 폴더에 저장되었습니다.")
    print("- 각 파일별 비교 그래프: [파일명]_comparison.png")
    print("- 전체 오차 통계: restoration_errors.csv")
    print("- 전체 오차 분포: error_distributions.png")


if __name__ == "__main__":
    process_files()
