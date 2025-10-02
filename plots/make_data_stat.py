import os
import json
import numpy as np
import pandas as pd
from glob import glob

def generate_feature_statistics(data_dir, output_path="data/feature_statistics.json"):
    """
    전체 데이터셋에서 각 피처별 통계 정보 계산

    Parameters:
    - data_dir: CSV 파일들이 있는 디렉토리 (예: "data/discharge/")
    - output_path: 저장할 JSON 파일 경로
    """

    # 모든 CSV 파일 경로 가져오기
    csv_files = glob(os.path.join(data_dir, "**/*.csv"), recursive=True)

    if not csv_files:
        print(f"{data_dir}에서 CSV 파일을 찾을 수 없습니다.")
        return

    print(f" 총 {len(csv_files)}개의 CSV 파일 발견")
    print(f" 통계 계산 중...")

    # 전체 데이터를 담을 딕셔너리
    all_data = {}

    # 모든 파일을 순회하며 데이터 수집
    for i, file_path in enumerate(csv_files, 1):
        if i % 100 == 0:
            print(f"   처리 중: {i}/{len(csv_files)}")

        try:
            df = pd.read_csv(file_path)

            # 각 컬럼(피처)별로 데이터 누적
            for col in df.columns:
                if col not in all_data:
                    all_data[col] = []

                # 결측치 제외하고 추가
                values = df[col].dropna().values
                all_data[col].extend(values)

        except Exception as e:
            print(f"⚠️  {file_path} 읽기 실패: {e}")
            continue

    # 통계 계산
    feature_stats = {}

    print(f"\n📈 피처별 통계 계산 중...")

    for feature, values in all_data.items():
        if len(values) == 0:
            continue

        values = np.array(values)

        # 이상치 제거 (옵션: z-score > 7 제거)
        # z_scores = np.abs((values - np.mean(values)) / np.std(values))
        # values = values[z_scores < 7]

        feature_stats[feature] = {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "variance": float(np.var(values)),
            "median": float(np.median(values)),
            "q1": float(np.percentile(values, 25)),
            "q3": float(np.percentile(values, 75)),
            "count": int(len(values))
        }

        print(f"   ✓ {feature}: mean={feature_stats[feature]['mean']:.4f}, "
              f"std={feature_stats[feature]['std']:.4f}")

    # JSON 저장
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(feature_stats, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 통계 정보 저장 완료: {output_path}")
    print(f"📊 총 {len(feature_stats)}개 피처 처리됨")

    return feature_stats


def verify_statistics(stats_path="data/feature_statistics.json"):
    """
    생성된 통계 정보 검증
    """
    with open(stats_path, 'r', encoding='utf-8') as f:
        stats = json.load(f)

    print("\n" + "="*80)
    print("피처 통계 정보 검증")
    print("="*80)

    for feature, info in stats.items():
        print(f"\n{feature}:")
        print(f"  Range: [{info['min']:.4f}, {info['max']:.4f}]")
        print(f"  Mean ± Std: {info['mean']:.4f} ± {info['std']:.4f}")
        print(f"  Variance: {info['variance']:.6f}")
        print(f"  Median: {info['median']:.4f}")
        print(f"  IQR: [{info['q1']:.4f}, {info['q3']:.4f}]")
        print(f"  Sample Count: {info['count']:,}")


# 원본 데이터에서 직접 계산 (전처리 전)
def calculate_from_original_data():
    """
    원본 NASA Battery Dataset에서 통계 계산
    """
    data_dir = "cycle_preprocess/csv/outlier_cut/threshold_7/cycle_len_512/"  # 실제 경로로 변경
    output_path = "data/feature_statistics.json"

    stats = generate_feature_statistics(data_dir, output_path)

    if stats:
        verify_statistics(output_path)

    return stats


# 학습 데이터에서 계산 (전처리 후)
def calculate_from_preprocessed_data(train_data_path):
    """
    전처리된 학습 데이터에서 통계 계산

    Parameters:
    - train_data_path: 전처리된 train 데이터 경로 (예: "preprocessed/train/")
    """
    output_path = "data/feature_statistics_preprocessed.json"

    stats = generate_feature_statistics(train_data_path, output_path)

    if stats:
        verify_statistics(output_path)

    return stats


if __name__ == "__main__":
    # 방법 1: 원본 데이터에서 계산 (권장)
    print("=" * 80)
    print("방법 1: 원본 데이터에서 통계 계산")
    print("=" * 80)
    calculate_from_original_data()

    # 방법 2: 전처리된 데이터에서 계산
    # print("\n" + "=" * 80)
    # print("방법 2: 전처리된 데이터에서 통계 계산")
    # print("=" * 80)
    # calculate_from_preprocessed_data("preprocessed/train/")
