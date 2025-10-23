import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

class CycleDataComparator:
    def __init__(self, original_path, reconstructed_path, threshold_percent=5):
        """
        Parameters:
        -----------
        original_path : str
            이상치 제거된 원본 CSV 경로
        reconstructed_path : str
            복원된 CSV 경로
        threshold_percent : float
            비교 기준 퍼센트 (5, 10, 15 등)
        """
        self.original_path = Path(original_path)
        self.reconstructed_path = Path(reconstructed_path)
        self.threshold_percent = threshold_percent

        # 비교할 피쳐 리스트
        self.features = [
            'Voltage_measured',
            'Current_measured',
            'Temperature_measured',
            'Current_load',
            'Voltage_load'
        ]

        # 결과 저장용
        self.results = {}

    def get_reconstructed_files(self):
        """복원본 CSV 파일 목록 가져오기"""
        reconstructed_files = sorted(self.reconstructed_path.glob('*_reconstructed.csv'))
        return reconstructed_files

    def extract_file_number(self, file_path):
        """파일명에서 번호 추출 (예: 00007_reconstructed.csv -> 00007)"""
        file_name = file_path.stem
        if '_reconstructed' in file_name:
            return file_name.split('_reconstructed')[0]
        return file_name

    def load_data_pair(self, reconstructed_file):
        """복원본과 원본 데이터 쌍 로드"""
        # 파일 번호 추출
        file_number = self.extract_file_number(reconstructed_file)

        # 원본 파일 경로
        original_file = self.original_path / f"{file_number}.csv"

        if not original_file.exists():
            print(f"Warning: Original file not found - {original_file}")
            return None, None, file_number

        # 데이터 로드
        try:
            original_df = pd.read_csv(original_file)
            reconstructed_df = pd.read_csv(reconstructed_file)
            return original_df, reconstructed_df, file_number
        except Exception as e:
            print(f"Error loading {file_number}: {e}")
            return None, None, file_number

    def calculate_feature_mean(self, df, feature):
        """특정 피쳐의 평균 계산"""
        if feature not in df.columns:
            print(f"Warning: Feature {feature} not found in dataframe")
            return None
        return df[feature].mean()

    def compare_data(self):
        """모든 데이터 쌍 비교 및 분석"""
        reconstructed_files = self.get_reconstructed_files()
        print(f"Found {len(reconstructed_files)} reconstructed files")
        print(f"Threshold: {self.threshold_percent}%\n")

        # 진행 상황 표시
        for reconstructed_file in tqdm(reconstructed_files, desc="Processing files"):
            original_df, reconstructed_df, file_number = self.load_data_pair(reconstructed_file)

            if original_df is None or reconstructed_df is None:
                continue

            # 행 개수 확인
            n_rows = len(original_df)
            if len(reconstructed_df) != n_rows:
                print(f"Warning: Row count mismatch in {file_number}")
                n_rows = min(len(original_df), len(reconstructed_df))

            # 결과 저장용 DataFrame 초기화 - 전체 및 피쳐별
            result_df = pd.DataFrame({
                'row_index': range(n_rows),
                'prediction_success': [True] * n_rows
            })

            # 피쳐별 결과 저장용 DataFrame 초기화
            feature_result_df = pd.DataFrame({
                'row_index': range(n_rows)
            })

            for feature in self.features:
                feature_result_df[f'{feature}_success'] = True

            # 각 피쳐별로 비교
            for feature in self.features:
                if feature not in original_df.columns or feature not in reconstructed_df.columns:
                    print(f"Warning: Feature {feature} not found in {file_number}")
                    continue

                # 원본 데이터의 피쳐 평균 계산
                feature_mean = self.calculate_feature_mean(original_df, feature)

                if feature_mean is None or feature_mean == 0:
                    continue

                # 임계값 계산 (평균의 threshold_percent%)
                threshold = abs(feature_mean) * (self.threshold_percent / 100)

                # 각 행별로 비교
                for idx in range(n_rows):
                    original_value = original_df.loc[idx, feature]
                    reconstructed_value = reconstructed_df.loc[idx, feature]

                    # 차이가 임계값 이상이면 False
                    diff = abs(original_value - reconstructed_value)
                    if diff > threshold:
                        result_df.loc[idx, 'prediction_success'] = False
                        feature_result_df.loc[idx, f'{feature}_success'] = False

            # 결과 저장 (전체 및 피쳐별)
            self.results[file_number] = {
                'overall': result_df,
                'by_feature': feature_result_df
            }

    def get_false_counts(self):
        """각 파일별 False 개수 집계"""
        false_counts = {}
        total_false = 0
        total_rows = 0

        for file_number, result_data in self.results.items():
            result_df = result_data['overall']
            false_count = (~result_df['prediction_success']).sum()
            total_count = len(result_df)
            false_counts[file_number] = {
                'false_count': false_count,
                'total_count': total_count,
                'false_percentage': (false_count / total_count * 100) if total_count > 0 else 0
            }
            total_false += false_count
            total_rows += total_count

        return false_counts, total_false, total_rows

    def get_feature_wise_false_counts(self):
        """피쳐별 False 개수 집계"""
        feature_wise_data = []

        for file_number, result_data in sorted(self.results.items()):
            feature_df = result_data['by_feature']
            total_count = len(feature_df)

            # 각 파일의 피쳐별 False 개수 계산
            row_data = {
                'file_number': file_number,
                'total_rows': total_count
            }

            # 전체 False 개수 (하나라도 False면 False인 행의 개수)
            overall_false = (~result_data['overall']['prediction_success']).sum()
            row_data['overall_false'] = overall_false
            row_data['overall_false_percentage'] = (overall_false / total_count * 100) if total_count > 0 else 0

            # 각 피쳐별 False 개수
            for feature in self.features:
                feature_col = f'{feature}_success'
                if feature_col in feature_df.columns:
                    false_count = (~feature_df[feature_col]).sum()
                    row_data[f'{feature}_false'] = false_count
                    row_data[f'{feature}_false_percentage'] = (false_count / total_count * 100) if total_count > 0 else 0
                else:
                    row_data[f'{feature}_false'] = 0
                    row_data[f'{feature}_false_percentage'] = 0.0

            feature_wise_data.append(row_data)

        return pd.DataFrame(feature_wise_data)

    def print_summary(self):
        """결과 요약 출력"""
        false_counts, total_false, total_rows = self.get_false_counts()

        print("\n" + "="*80)
        print(f"SUMMARY - Threshold: {self.threshold_percent}%")
        print("="*80)
        print(f"\nTotal files processed: {len(self.results)}")
        print(f"Total rows analyzed: {total_rows}")
        print(f"Total prediction failures (False): {total_false}")
        print(f"Overall failure rate: {(total_false/total_rows*100):.2f}%")

        print("\n" + "-"*80)
        print("Per-file False counts:")
        print("-"*80)

        # 파일별 정렬하여 출력
        sorted_files = sorted(false_counts.items())
        for file_number, counts in sorted_files:
            print(f"File {file_number}: {counts['false_count']:4d} / {counts['total_count']:4d} "
                  f"({counts['false_percentage']:6.2f}%)")

        print("="*80)

    def print_feature_wise_summary(self, save_path=None):
        """피쳐별 False 개수 요약 출력"""
        feature_df = self.get_feature_wise_false_counts()

        # 출력 내용을 문자열로 저장
        summary_lines = []
        summary_lines.append("\n" + "="*80)
        summary_lines.append(f"FEATURE-WISE SUMMARY - Threshold: {self.threshold_percent}%")
        summary_lines.append("="*80)

        # 피쳐별 전체 False 개수 계산
        summary_lines.append("\nTotal False counts by feature:")
        summary_lines.append("-"*80)
        for feature in self.features:
            total_false = feature_df[f'{feature}_false'].sum()
            total_rows = feature_df['total_rows'].sum()
            percentage = (total_false / total_rows * 100) if total_rows > 0 else 0
            line = f"{feature:30s}: {total_false:6d} / {total_rows:6d} ({percentage:6.2f}%)"
            summary_lines.append(line)

        summary_lines.append("\nOverall (any feature failed):")
        summary_lines.append("-"*80)
        total_overall_false = feature_df['overall_false'].sum()
        total_rows = feature_df['total_rows'].sum()
        percentage = (total_overall_false / total_rows * 100) if total_rows > 0 else 0
        line = f"{'Overall':30s}: {total_overall_false:6d} / {total_rows:6d} ({percentage:6.2f}%)"
        summary_lines.append(line)
        summary_lines.append("="*80)

        # 콘솔 출력
        for line in summary_lines:
            print(line)

        # 파일 저장
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(summary_lines))
            print(f"\nFeature-wise summary saved to: {save_path}")

    def save_results(self, output_path):
        """결과를 CSV 파일로 저장"""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # csv_results 디렉토리 먼저 생성 (반복문 밖에서 한 번만)
        csv_results_path = output_path / "csv_results"
        csv_results_path.mkdir(parents=True, exist_ok=True)

        # 각 파일별 결과 저장
        for file_number, result_data in self.results.items():
            # 피쳐별 결과 저장
            feature_output_file = csv_results_path / f"{file_number}_feature_wise_result.csv"
            result_data['by_feature'].to_csv(feature_output_file, index=False)

        # 전체 요약 저장 (기존 방식)
        false_counts, total_false, total_rows = self.get_false_counts()
        summary_data = []
        for file_number, counts in sorted(false_counts.items()):
            summary_data.append({
                'file_number': file_number,
                'false_count': counts['false_count'],
                'total_count': counts['total_count'],
                'false_percentage': counts['false_percentage']
            })

        summary_df = pd.DataFrame(summary_data)
        summary_file = output_path / f"summary_threshold_{self.threshold_percent}percent.csv"
        summary_df.to_csv(summary_file, index=False)

        # 피쳐별 False 개수 요약 저장
        feature_wise_df = self.get_feature_wise_false_counts()
        feature_wise_file = output_path / f"feature_wise_summary_threshold_{self.threshold_percent}percent.csv"
        feature_wise_df.to_csv(feature_wise_file, index=False)

        print(f"\nResults saved to: {output_path}")
        print(f"  - Overall summary: {summary_file.name}")
        print(f"  - Feature-wise summary: {feature_wise_file.name}")
        print(f"  - Individual CSV files in: {csv_results_path}")

        # 텍스트 요약 저장
        summary_text_file = output_path / f"feature_wise_text_summary_threshold_{self.threshold_percent}percent.txt"
        return summary_text_file  # main에서 사용할 수 있도록 반환


def main():
    case_number = "46.2.5"
    # model_type = "deepsc"
    model_type = "lstm"
    # 임계값 설정 (5%, 10%, 15% 등)
    threshold_percent = 15  # 이 값을 변경하여 다른 임계값 사용 가능

    # 경로 설정
    original_path = r"./cycle_preprocess/csv/outlier_cut/threshold_7/cycle_len_512"
    reconstructed_path = f"./reconstruction/case{case_number}/reconstructed_rayleigh_{model_type}_MSE"

    # 비교 실행
    comparator = CycleDataComparator(
        original_path=original_path,
        reconstructed_path=reconstructed_path,
        threshold_percent=threshold_percent
    )

    print("Starting comparison...")
    comparator.compare_data()

    # 결과 출력
    comparator.print_summary()

    # 결과 저장
    output_path = f"./comparison_results/case_{case_number}/threshold_{threshold_percent}percent"
    summary_text_path = comparator.save_results(output_path)

    # 피쳐별 요약을 텍스트 파일로 저장
    comparator.print_feature_wise_summary(save_path=summary_text_path)

    # 여러 임계값으로 실행하고 싶다면:
    # print("\n" + "="*80)
    # print("Running comparison with multiple thresholds...")
    # print("="*80)

    # for threshold in [5, 10, 15]:
    #     print(f"\n\nAnalyzing with {threshold}% threshold...")
    #     comparator = CycleDataComparator(
    #         original_path=original_path,
    #         reconstructed_path=reconstructed_path,
    #         threshold_percent=threshold
    #     )
    #     comparator.compare_data()
    #     comparator.print_summary()
    #     comparator.print_feature_wise_summary()
    #     comparator.save_results(f"./comparison_results/threshold_{threshold}percent")


if __name__ == "__main__":
    main()
