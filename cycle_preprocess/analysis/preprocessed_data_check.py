import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pdb


def plot_dataset_comparison(data1_path, data2_path, save_path="comparison_plots"):
    """
    두 개의 train_data.pt 파일을 로드하여 비교 시각화
    """
    # 데이터 로드
    data1 = torch.load(data1_path)
    data2 = torch.load(data2_path)

    data1 = data1.tensors[0]  # (batch_size, seq_len, n_features)
    data2 = data2.tensors[0]  # (batch_size, seq_len, n_features)

    pdb.set_trace()

    print(f"Dataset 1 shape: {data1.shape}")
    print(f"Dataset 2 shape: {data2.shape}")

    # shape 출력 및 확인
    print("\n데이터 형태:")
    print(f"Dataset 1: {data1.shape} (배치 크기 x 시퀀스 길이 x 특성 수)")
    print(f"Dataset 2: {data2.shape} (배치 크기 x 시퀀스 길이 x 특성 수)")

    # numpy로 변환 (3D 텐서 유지)
    data1_np = data1.numpy()
    data2_np = data2.numpy()

    # 특성 수 확인
    n_features = data1_np.shape[2]  # 마지막 차원이 특성 수
    feature_names = [
        "Voltage_measured",
        "Current_measured",
        "Temperature_measured",
        "Current_load",
        "Voltage_load",
        "Time",
    ]

    # 1. 분포 비교 (각 특성별로 모든 시퀀스의 값을 사용)
    plt.figure(figsize=(15, 10))
    for i in range(n_features):
        plt.subplot(2, 3, i + 1)

        # 모든 시퀀스의 값을 flatten하여 히스토그램 생성
        values1 = data1_np[:, :, i].flatten()
        values2 = data2_np[:, :, i].flatten()

        plt.hist(values1, bins=50, alpha=0.5, label="Dataset 1", density=True)
        plt.hist(values2, bins=50, alpha=0.5, label="Dataset 2", density=True)

        plt.title(f"{feature_names[i]} Distribution")
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.legend()

    plt.tight_layout()
    plt.savefig(f"{save_path}/distribution_comparison.png")
    plt.close()

    # 2. Box Plot 비교
    plt.figure(figsize=(15, 6))
    box_data = [
        (data1_np[:, :, i].flatten(), data2_np[:, :, i].flatten())
        for i in range(n_features)
    ]

    plt.boxplot(
        [item for pair in box_data for item in pair],
        labels=[f"{name}\nSet {i+1}" for name in feature_names for i in range(2)],
    )

    plt.xticks(rotation=45)
    plt.title("Feature Distribution Comparison (Box Plot)")
    plt.tight_layout()
    plt.savefig(f"{save_path}/boxplot_comparison.png")
    plt.close()

    # 3. 시계열 샘플 비교 (첫 번째 배치의 시퀀스)
    plt.figure(figsize=(15, 10))

    for i in range(n_features):
        plt.subplot(2, 3, i + 1)

        # 첫 번째 배치의 시퀀스 데이터 플롯
        plt.plot(data1_np[0, :, i], alpha=0.7, label="Dataset 1")
        plt.plot(data2_np[0, :, i], alpha=0.7, label="Dataset 2")

        plt.title(f"{feature_names[i]} Time Series\n(First sequence)")
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.legend()

    plt.tight_layout()
    plt.savefig(f"{save_path}/timeseries_comparison.png")
    plt.close()

    # 4. 통계 비교
    print("\n=== 통계 비교 ===")
    for i in range(n_features):
        print(f"\n{feature_names[i]}:")
        # 모든 시퀀스의 값을 사용하여 통계 계산
        values1 = data1_np[:, :, i].flatten()
        values2 = data2_np[:, :, i].flatten()

        print(f"Dataset 1 - Mean: {values1.mean():.4f}, Std: {values1.std():.4f}")
        print(f"Dataset 2 - Mean: {values2.mean():.4f}, Std: {values2.std():.4f}")
        print(
            f"Absolute Difference in Mean: {abs(values1.mean() - values2.mean()):.4f}"
        )

        # 시퀀스별 통계도 추가
        print("시퀀스별 통계:")
        print(
            f"Dataset 1 - Mean per sequence: {data1_np[:, :, i].mean(axis=1).mean():.4f}"
        )
        print(
            f"Dataset 2 - Mean per sequence: {data2_np[:, :, i].mean(axis=1).mean():.4f}"
        )


if __name__ == "__main__":
    data1_path = "./cycle_preprocess/preprocessed/processed_minmax/train_data.pt"
    # data2_path = "./cycle_preprocess/preprocessed/processed_zscore/train_data.pt"
    data2_path = "./preprocessed/preprocessed_data_0717_outlier_cut/train_data.pt"
    save_path = "./cycle_preprocess/analysis/preprocessed_comparison_plots_2"

    parser = argparse.ArgumentParser(description="Compare two preprocessed datasets")
    parser.add_argument(
        "--data1",
        type=str,
        default=data1_path,
        help="Path to first train_data.pt",
    )
    parser.add_argument(
        "--data2",
        type=str,
        default=data2_path,
        help="Path to second train_data.pt",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=save_path,
        help="Directory to save comparison plots",
    )

    args = parser.parse_args()

    # 저장 디렉토리 생성
    import os

    os.makedirs(args.save_path, exist_ok=True)

    # 데이터셋 비교
    plot_dataset_comparison(args.data1, args.data2, args.save_path)

    print(f"\n비교 결과가 {args.save_path} 폴더에 저장되었습니다.")

# 사용 예시:
# python preprocessed_data_check.py --data1 cycle_preprocess/preprocessed/processed_minmax/train_data.pt --data2 cycle_preprocess/preprocessed/processed_zscore/train_data.pt
