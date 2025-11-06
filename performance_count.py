import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pdb

class CycleDataComparator:
    # ✨ 수정: threshold_method 파라미터 추가 (문자열로 방식 지정)
    def __init__(self, original_path, reconstructed_path, threshold_percent=5, threshold_method="mean", min_absolute_threshold=1e-9):
        """
        Parameters:
        -----------
        original_path : str
            이상치 제거된 원본 CSV 경로
        reconstructed_path : str
            복원된 CSV 경로
        threshold_percent : float
            비교 기준 퍼센트 (5, 10, 15 등)
        threshold_method : str, optional
            임계값 계산 방식: "mean", "range_center", "point_wise" (default: "mean")
        min_absolute_threshold : float, optional
            "point_wise" 방식에서 원본 값이 0에 가까울 때 사용할 최소 절대 임계값 (default: 1e-9)
        """
        if threshold_method not in ["mean", "range_center", "point_wise"]:
            raise ValueError("threshold_method must be one of 'mean', 'range_center', or 'point_wise'")

        self.original_path = Path(original_path)
        self.reconstructed_path = Path(reconstructed_path)
        self.threshold_percent = threshold_percent
        self.threshold_method = threshold_method
        self.min_absolute_threshold = min_absolute_threshold # point_wise용 최소 절대 임계값
        self.threshold_ratio = self.threshold_percent / 100.0 # 미리 계산

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
        """파일명에서 번호 추출"""
        file_name = file_path.stem
        if '_reconstructed' in file_name:
            return file_name.split('_reconstructed')[0]
        return file_name

    def load_data_pair(self, reconstructed_file):
        """복원본과 원본 데이터 쌍 로드"""
        file_number = self.extract_file_number(reconstructed_file)
        original_file = self.original_path / f"{file_number}.csv"

        if not original_file.exists():
            print(f"Warning: Original file not found - {original_file}")
            return None, None, file_number

        try:
            original_df = pd.read_csv(original_file)
            reconstructed_df = pd.read_csv(reconstructed_file)
            return original_df, reconstructed_df, file_number
        except Exception as e:
            print(f"Error loading {file_number}: {e}")
            return None, None, file_number

    def calculate_feature_mean(self, df, feature):
        """피쳐 평균 계산"""
        if feature not in df.columns: return None
        return df[feature].mean()

    def calculate_feature_range_center(self, df, feature):
        """피쳐 Min/Max 평균 계산"""
        if feature not in df.columns: return None
        min_val = df[feature].min()
        max_val = df[feature].max()
        return (min_val + max_val) / 2

    def compare_data(self):
        """모든 데이터 쌍 비교 및 분석 (선택된 threshold_method 사용)"""
        reconstructed_files = self.get_reconstructed_files()
        print(f"Found {len(reconstructed_files)} reconstructed files")
        print(f"Threshold: {self.threshold_percent}% based on {self.threshold_method}")
        print(f"\n")

        for reconstructed_file in tqdm(reconstructed_files, desc=f"Processing files ({self.threshold_method} threshold)"):
            original_df, reconstructed_df, file_number = self.load_data_pair(reconstructed_file)

            if original_df is None or reconstructed_df is None:
                continue

            n_rows = len(original_df)
            if len(reconstructed_df) != n_rows:
                print(f"Warning: Row count mismatch in {file_number} (Original: {n_rows}, Reconstructed: {len(reconstructed_df)}). Using min length.")
                n_rows = min(n_rows, len(reconstructed_df))
                original_df = original_df.iloc[:n_rows]
                reconstructed_df = reconstructed_df.iloc[:n_rows]


            result_df = pd.DataFrame({'row_index': range(n_rows), 'prediction_success': [True] * n_rows})
            feature_result_df = pd.DataFrame({'row_index': range(n_rows)})
            for feature in self.features:
                feature_result_df[f'{feature}_success'] = True

            for feature in self.features:
                if feature not in original_df.columns or feature not in reconstructed_df.columns:
                    print(f"Warning: Feature {feature} not found in {file_number}")
                    feature_result_df[f'{feature}_success'] = False # 피쳐 없으면 전체 False 처리
                    result_df['prediction_success'] = False # 전체 결과에도 영향
                    continue

                # 임계값 계산 (방식별 분기)
                absolute_threshold = None # mean, range_center 용
                if self.threshold_method in ["mean", "range_center"]:
                    reference_value = None
                    if self.threshold_method == "mean":
                        reference_value = self.calculate_feature_mean(original_df, feature)
                    else: # "range_center"
                        reference_value = self.calculate_feature_range_center(original_df, feature)

                    if reference_value is None:
                        print(f"Warning: Could not calculate {self.threshold_method} reference for {feature} in {file_number}. Marking as failed.")
                        feature_result_df[f'{feature}_success'] = False
                        result_df['prediction_success'] = False
                        continue

                    absolute_threshold = abs(reference_value) * self.threshold_ratio if reference_value != 0 else 0


                # 각 행별로 비교
                for idx in range(n_rows):
                    original_value = original_df.loc[idx, feature]
                    reconstructed_value = reconstructed_df.loc[idx, feature]

                    # NaN 값 처리: 하나라도 NaN이면 False
                    if pd.isna(original_value) or pd.isna(reconstructed_value):
                         is_false = True
                    else:
                        diff = abs(original_value - reconstructed_value)
                        is_false = False

                        # 실패 조건 판정 (방식별 분기)
                        if self.threshold_method == "point_wise":
                            # 원본 값이 0에 가까운 경우: 최소 절대 임계값 사용
                            if abs(original_value) < self.min_absolute_threshold:
                                if diff > self.min_absolute_threshold:
                                    is_false = True
                            # 원본 값이 0이 아닌 경우: 상대 임계값 사용
                            else:
                                relative_threshold = abs(original_value) * self.threshold_ratio
                                if diff > relative_threshold:
                                    is_false = True
                        else: # "mean" or "range_center"
                            # 미리 계산된 절대 임계값 사용
                            if absolute_threshold is not None and diff > absolute_threshold:
                                is_false = True

                    # False 판정 시 결과 업데이트
                    if is_false:
                        result_df.loc[idx, 'prediction_success'] = False
                        feature_result_df.loc[idx, f'{feature}_success'] = False


            # 결과 저장
            self.results[file_number] = {
                'overall': result_df,
                'by_feature': feature_result_df
            }

    # get_false_counts, get_feature_wise_false_counts 함수는 수정 없이 사용 가능

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
        """피쳐별 False 개수 집계 (DataFrame 반환)"""
        feature_wise_data = []

        for file_number, result_data in sorted(self.results.items()):
            feature_df = result_data['by_feature']
            total_count = len(feature_df)

            row_data = {'file_number': file_number, 'total_rows': total_count}

            overall_false = (~result_data['overall']['prediction_success']).sum()
            row_data['overall_false'] = overall_false
            row_data['overall_false_percentage'] = (overall_false / total_count * 100) if total_count > 0 else 0

            for feature in self.features:
                feature_col = f'{feature}_success'
                if feature_col in feature_df.columns:
                    false_count = (~feature_df[feature_col]).sum()
                    row_data[f'{feature}_false'] = false_count
                    row_data[f'{feature}_false_percentage'] = (false_count / total_count * 100) if total_count > 0 else 0
                else:
                    row_data[f'{feature}_false'] = np.nan
                    row_data[f'{feature}_false_percentage'] = np.nan

            feature_wise_data.append(row_data)

        return pd.DataFrame(feature_wise_data)


    # 수정: 요약 출력 시 threshold_method 명시
    def print_summary(self):
        """결과 요약 출력"""
        false_counts, total_false, total_rows = self.get_false_counts()

        print("\n" + "="*80)
        print(f"SUMMARY - Threshold: {self.threshold_percent}% (based on {self.threshold_method})")
        print("="*80)
        # ... (이하 동일)
        print(f"\nTotal files processed: {len(self.results)}")
        print(f"Total rows analyzed: {total_rows}")
        print(f"Total prediction failures (False): {total_false}")
        if total_rows > 0:
            print(f"Overall failure rate: {(total_false/total_rows*100):.2f}%")
        else:
             print("Overall failure rate: N/A (no rows analyzed)")


        print("\n" + "-"*80)
        print("Per-file False counts:")
        print("-"*80)

        sorted_files = sorted(false_counts.items())
        for file_number, counts in sorted_files:
            print(f"File {file_number}: {counts['false_count']:4d} / {counts['total_count']:4d} "
                  f"({counts['false_percentage']:6.2f}%)")

        print("="*80)

    # 수정: 요약 출력 시 threshold_method 명시
    def print_feature_wise_summary(self, save_path=None):
        """피쳐별 False 개수 요약 출력 및 저장"""
        feature_df = self.get_feature_wise_false_counts()

        if feature_df.empty:
            print("No feature-wise data to summarize.")
            return

        summary_lines = []
        summary_lines.append("\n" + "="*80)
        summary_lines.append(f"FEATURE-WISE SUMMARY - Threshold: {self.threshold_percent}% (based on {self.threshold_method})")
        summary_lines.append("="*80)
        # ... (이하 동일)
        summary_lines.append("\nTotal False counts by feature:")
        summary_lines.append("-"*80)
        total_rows_overall = feature_df['total_rows'].sum()

        for feature in self.features:
            false_col = f'{feature}_false'
            if false_col in feature_df.columns and not feature_df[false_col].isnull().all(): # NaN 체크 추가
                total_false = feature_df[false_col].sum()
                percentage = (total_false / total_rows_overall * 100) if total_rows_overall > 0 else 0
                line = f"{feature:30s}: {int(total_false):6d} / {total_rows_overall:6d} ({percentage:6.2f}%)"
            else:
                 line = f"{feature:30s}: Data N/A"
            summary_lines.append(line)


        summary_lines.append("\nOverall (any feature failed):")
        summary_lines.append("-"*80)
        if 'overall_false' in feature_df.columns and not feature_df['overall_false'].isnull().all(): # NaN 체크 추가
            total_overall_false = feature_df['overall_false'].sum()
            percentage = (total_overall_false / total_rows_overall * 100) if total_rows_overall > 0 else 0
            line = f"{'Overall':30s}: {int(total_overall_false):6d} / {total_rows_overall:6d} ({percentage:6.2f}%)"
        else:
             line = f"{'Overall':30s}: Data N/A"
        summary_lines.append(line)
        summary_lines.append("="*80)

        for line in summary_lines:
            print(line)

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(summary_lines))
                print(f"\nFeature-wise summary saved to: {save_path}")
            except Exception as e:
                print(f"Error saving feature-wise summary to {save_path}: {e}")

    # 수정: 저장 경로에 threshold_method 포함
    def save_results(self, output_path):
        """결과를 CSV 파일로 저장"""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # threshold 방법 폴더 생성 (더 명확한 이름)
        threshold_method_folder = f"threshold_{self.threshold_percent}percent_{self.threshold_method}"
        method_output_path = output_path / threshold_method_folder
        method_output_path.mkdir(parents=True, exist_ok=True)

        csv_results_path = method_output_path / "csv_results"
        csv_results_path.mkdir(parents=True, exist_ok=True)

        for file_number, result_data in self.results.items():
            feature_output_file = csv_results_path / f"{file_number}_feature_wise_result.csv"
            result_data['by_feature'].to_csv(feature_output_file, index=False)

        false_counts, _, _ = self.get_false_counts()
        summary_data = []
        for file_number, counts in sorted(false_counts.items()):
            summary_data.append({
                'file_number': file_number,
                'false_count': counts['false_count'],
                'total_count': counts['total_count'],
                'false_percentage': counts['false_percentage']
            })
        summary_df = pd.DataFrame(summary_data)
        summary_file = method_output_path / f"summary_per_file.csv"
        summary_df.to_csv(summary_file, index=False)

        feature_wise_df = self.get_feature_wise_false_counts()
        feature_wise_file = method_output_path / f"summary_feature_wise.csv"
        feature_wise_df.to_csv(feature_wise_file, index=False)

        print(f"\nResults saved to: {method_output_path}")
        print(f"  - Overall summary (per file): {summary_file.name}")
        print(f"  - Feature-wise summary: {feature_wise_file.name}")
        print(f"  - Individual CSV files in: {csv_results_path.name}/")

        summary_text_file = method_output_path / f"summary_feature_wise_text.txt"
        return summary_text_file


def main():
    case_number = "10002.1.8"
    model_type = "deepsc"
    threshold_percent = 15

    comparison_method = "mean"
    # comparison_method = "range_center"
    # comparison_method = "point_wise"

    min_abs_thresh = 1e-6 # 예시 값, 필요시 조정

    original_path = r"./cycle_preprocess/csv/outlier_cut/threshold_7/cycle_len_512"
    reconstructed_path = f"./reconstruction/case{case_number}/reconstructed_rayleigh_{model_type}_MSE"

    comparator = CycleDataComparator(
        original_path=original_path,
        reconstructed_path=reconstructed_path,
        threshold_percent=threshold_percent,
        threshold_method=comparison_method, # 선택된 방법 전달
        min_absolute_threshold=min_abs_thresh # point_wise용 임계값 전달
    )

    print("Starting comparison...")
    comparator.compare_data()
    comparator.print_summary()

    output_path_base = f"./comparison_results/case_{case_number}"
    summary_text_path = comparator.save_results(output_path_base)
    comparator.print_feature_wise_summary(save_path=summary_text_path)

    # 여러 임계값과 방법으로 실행하는 예제
    # print("\n" + "="*80)
    # print("Running comparison with multiple thresholds and methods...")
    # print("="*80)

    # for threshold in [5, 10, 15]:
    #     for method in ["mean", "range_center", "point_wise"]:
    #         print(f"\n\nAnalyzing with {threshold}% threshold (based on {method})...")
    #         comparator = CycleDataComparator(
    #             original_path=original_path,
    #             reconstructed_path=reconstructed_path,
    #             threshold_percent=threshold,
    #             threshold_method=method,
    #             min_absolute_threshold=min_abs_thresh # 항상 전달
    #         )
    #         comparator.compare_data()
    #         # comparator.print_summary() # 요약 출력은 선택적
    #         output_path_base = f"./comparison_results/case_{case_number}"
    #         summary_text_path = comparator.save_results(output_path_base)
    #         comparator.print_feature_wise_summary(save_path=summary_text_path)


if __name__ == "__main__":
    main()
