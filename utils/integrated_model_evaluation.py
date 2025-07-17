import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

save_dir = "./analysis/250708_B0047/"

def calculate_integrated_residuals():
    """
    3가지 손실함수 모델의 잔차를 통합 분석하여 모델 성능 평가
    """
    
    # 폴더 경로들
    folders = {
        'Huber': 'reconstructed_no_channel_deepsc_Huber',
        'MAE': 'reconstructed_no_channel_deepsc_MAE', 
        'MSE': 'reconstructed_no_channel_deepsc_MSE'
    }
    
    # 원본 데이터 경로
    original_data_path = 'data_handling/merged/B0047.csv'
    
    # 원본 데이터 로드
    print("원본 데이터 로딩 중...")
    original_data = pd.read_csv(original_data_path)
    
    # 6개 피쳐 컬럼들
    feature_columns = ['Voltage_measured', 'Current_measured', 'Temperature_measured', 
                      'Current_load', 'Voltage_load', 'Time']
    
    # 각 모델의 잔차와 성능 지표를 저장할 딕셔너리
    model_results = {}
    
    for model_name, folder in folders.items():
        reconstructed_file = f"{folder}/B0047_reconstructed.csv"
        
        if os.path.exists(reconstructed_file):
            print(f"\n{model_name} 모델 분석 중...")
            reconstructed_data = pd.read_csv(reconstructed_file)
            
            # 잔차 계산
            residuals = original_data[feature_columns] - reconstructed_data[feature_columns]
            
            # 성능 지표 계산
            performance_metrics = {}
            for col in feature_columns:
                # MSE
                mse = mean_squared_error(original_data[col], reconstructed_data[col])
                # MAE
                mae = mean_absolute_error(original_data[col], reconstructed_data[col])
                # RMSE
                rmse = np.sqrt(mse)
                # 상대 오차 (MAPE)
                mape = np.mean(np.abs(residuals[col] / original_data[col])) * 100
                
                performance_metrics[col] = {
                    'MSE': mse,
                    'MAE': mae,
                    'RMSE': rmse,
                    'MAPE': mape
                }
            
            # z-score 표준화된 잔차
            residuals_zscore = pd.DataFrame()
            for col in feature_columns:
                residuals_zscore[col] = stats.zscore(residuals[col], nan_policy='omit')
            
            model_results[model_name] = {
                'residuals': residuals,
                'residuals_zscore': residuals_zscore,
                'performance': performance_metrics
            }
            
            print(f"{model_name} 모델 분석 완료")
        else:
            print(f"경고: {reconstructed_file} 파일을 찾을 수 없습니다.")
    
    if not model_results:
        print("오류: 처리할 수 있는 모델 데이터가 없습니다.")
        return None
    
    return model_results, original_data, feature_columns

def calculate_integrated_metrics(model_results, feature_columns):
    """
    통합 평가 지표 계산
    """
    
    # 1. 평균 잔차 (z-score 표준화)
    all_zscore_residuals = []
    for model_name, results in model_results.items():
        all_zscore_residuals.append(results['residuals_zscore'])
    
    
    mean_residuals = pd.concat(all_zscore_residuals).groupby(level=0).mean()
    
    # 2. 모델 간 일관성 지표
    consistency_metrics = {}
    for col in feature_columns:
        # 3개 모델의 잔차 표준편차 (일관성 측정)
        model_residuals = [results['residuals_zscore'][col] for results in model_results.values()]
        consistency = np.std(model_residuals, axis=0).mean()
        consistency_metrics[col] = consistency
    
    # 3. 통합 성능 점수
    integrated_scores = {}
    for col in feature_columns:
        # 각 모델의 성능 지표 평균
        mse_scores = [results['performance'][col]['MSE'] for results in model_results.values()]
        mae_scores = [results['performance'][col]['MAE'] for results in model_results.values()]
        rmse_scores = [results['performance'][col]['RMSE'] for results in model_results.values()]
        
        integrated_scores[col] = {
            'Mean_MSE': np.mean(mse_scores),
            'Mean_MAE': np.mean(mae_scores),
            'Mean_RMSE': np.mean(rmse_scores),
            'Consistency': consistency_metrics[col]
        }
    
    return mean_residuals, consistency_metrics, integrated_scores

def create_comprehensive_visualization(model_results, mean_residuals, feature_columns):
    """
    종합적인 시각화 생성
    """
    
    # 1. 모델별 성능 비교
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, col in enumerate(feature_columns):
        mse_values = [results['performance'][col]['MSE'] for results in model_results.values()]
        mae_values = [results['performance'][col]['MAE'] for results in model_results.values()]
        
        x = np.arange(len(model_results))
        width = 0.35
        
        axes[i].bar(x - width/2, mse_values, width, label='MSE', alpha=0.8)
        axes[i].bar(x + width/2, mae_values, width, label='MAE', alpha=0.8)
        
        axes[i].set_title(f'{col} - 모델별 성능 비교')
        axes[i].set_ylabel('오차')
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(list(model_results.keys()))
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir+'B0047_model_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. 통합 잔차 시계열
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(feature_columns, 1):
        plt.subplot(2, 3, i)
        plt.plot(mean_residuals[col], alpha=0.7, linewidth=1)
        plt.title(f'{col} - 통합 잔차 (Z-score)')
        plt.xlabel('시간 인덱스')
        plt.ylabel('표준화된 잔차')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir+'B0047_integrated_residuals_timeseries.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. 모델 간 잔차 분포 비교
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, col in enumerate(feature_columns):
        for model_name, results in model_results.items():
            axes[i].hist(results['residuals_zscore'][col], bins=30, alpha=0.6, 
                        label=model_name, density=True)
        
        axes[i].set_title(f'{col} - 모델별 잔차 분포')
        axes[i].set_xlabel('Z-score 표준화된 잔차')
        axes[i].set_ylabel('밀도')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir+'B0047_model_residual_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_integrated_results(mean_residuals, integrated_scores, model_results):
    """
    통합 결과 저장
    """
    
    # 1. 평균 잔차 저장
    mean_residuals.to_csv(save_dir+'B0047_integrated_residuals.csv', index=False)
    
    # 2. 통합 성능 지표 저장
    integrated_df = pd.DataFrame(integrated_scores).T
    integrated_df.to_csv(save_dir+'B0047_integrated_performance_metrics.csv')
    
    # 3. 상세 성능 보고서 생성
    with open(save_dir+'B0047_integrated_evaluation_report.txt', 'w', encoding='utf-8') as f:
        f.write("=== B0047 통합 모델 평가 보고서 ===\n\n")
        
        f.write("1. 모델별 개별 성능 지표:\n")
        f.write("-" * 50 + "\n")
        
        for model_name, results in model_results.items():
            f.write(f"\n{model_name} 모델:\n")
            
            # 각 피쳐별 성능 지표
            for col in ['Voltage_measured', 'Current_measured', 'Temperature_measured', 
                       'Current_load', 'Voltage_load', 'Time']:
                perf = results['performance'][col]
                f.write(f"  {col}:\n")
                f.write(f"    MSE: {perf['MSE']:.6f}\n")
                f.write(f"    MAE: {perf['MAE']:.6f}\n")
                f.write(f"    RMSE: {perf['RMSE']:.6f}\n")
                f.write(f"    MAPE: {perf['MAPE']:.2f}%\n")
            
            # 모델별 모든 피쳐의 평균 성능 지표
            avg_mse = np.mean([results['performance'][col]['MSE'] for col in ['Voltage_measured', 'Current_measured', 'Temperature_measured', 
                       'Current_load', 'Voltage_load', 'Time']])
            avg_mae = np.mean([results['performance'][col]['MAE'] for col in ['Voltage_measured', 'Current_measured', 'Temperature_measured', 
                       'Current_load', 'Voltage_load', 'Time']])
            avg_rmse = np.mean([results['performance'][col]['RMSE'] for col in ['Voltage_measured', 'Current_measured', 'Temperature_measured', 
                       'Current_load', 'Voltage_load', 'Time']])
            
            f.write(f"  전체 피쳐 평균:\n")
            f.write(f"    평균 MSE: {avg_mse:.6f}\n")
            f.write(f"    평균 MAE: {avg_mae:.6f}\n")
            f.write(f"    평균 RMSE: {avg_rmse:.6f}\n")
        
        f.write("\n\n2. 통합 성능 지표:\n")
        f.write("-" * 50 + "\n")
        for col, scores in integrated_scores.items():
            f.write(f"\n{col}:\n")
            f.write(f"  평균 MSE: {scores['Mean_MSE']:.6f}\n")
            f.write(f"  평균 MAE: {scores['Mean_MAE']:.6f}\n")
            f.write(f"  평균 RMSE: {scores['Mean_RMSE']:.6f}\n")
            f.write(f"  모델 일관성: {scores['Consistency']:.6f}\n")
        
        f.write("\n\n3. 통합 평가 결론:\n")
        f.write("-" * 50 + "\n")
        
        # 최고 성능 모델 찾기
        overall_mse = []
        for model_name, results in model_results.items():
            avg_mse = np.mean([results['performance'][col]['MSE'] for col in ['Voltage_measured', 'Current_measured', 'Temperature_measured', 
                       'Current_load', 'Voltage_load', 'Time']])
            overall_mse.append((model_name, avg_mse))
        
        best_model = min(overall_mse, key=lambda x: x[1])
        f.write(f"전체적으로 가장 좋은 성능을 보인 모델: {best_model[0]} (평균 MSE: {best_model[1]:.6f})\n")
        
        # 일관성 분석
        consistency_values = [scores['Consistency'] for scores in integrated_scores.values()]
        avg_consistency = np.mean(consistency_values)
        f.write(f"모델 간 평균 일관성: {avg_consistency:.6f}\n")
        f.write("(값이 낮을수록 모델들이 일관된 결과를 보임)\n")

def main():
    """
    메인 실행 함수
    """
    print("=== B0047 통합 모델 평가 시작 ===\n")
    
    # 데이터 로드 및 분석
    results = calculate_integrated_residuals()
    if results is None:
        return
    
    model_results, original_data, feature_columns = results
    
    # 통합 지표 계산
    print("\n통합 평가 지표 계산 중...")
    mean_residuals, consistency_metrics, integrated_scores = calculate_integrated_metrics(
        model_results, feature_columns
    )
    
    # 시각화
    print("시각화 생성 중...")
    create_comprehensive_visualization(model_results, mean_residuals, feature_columns)
    
    # 결과 저장
    print("결과 저장 중...")
    save_integrated_results(mean_residuals, integrated_scores, model_results)
    
    # 요약 출력
    print("\n=== 통합 평가 완료 ===")
    print("생성된 파일들:")
    print("- B0047_integrated_residuals.csv: 통합 평균 잔차")
    print("- B0047_integrated_performance_metrics.csv: 통합 성능 지표")
    print("- B0047_integrated_evaluation_report.txt: 상세 평가 보고서")
    print("- B0047_model_performance_comparison.png: 모델별 성능 비교")
    print("- B0047_integrated_residuals_timeseries.png: 통합 잔차 시계열")
    print("- B0047_model_residual_distributions.png: 모델별 잔차 분포")
    
    # 간단한 요약 출력
    print("\n=== 주요 결과 요약 ===")
    for col in feature_columns:
        scores = integrated_scores[col]
        print(f"{col}:")
        print(f"  평균 RMSE: {scores['Mean_RMSE']:.6f}")
        print(f"  모델 일관성: {scores['Consistency']:.6f}")

if __name__ == "__main__":
    main() 