import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

save_dir = "./analysis/250708_B0047/" 

def calculate_residual_mean():
    """
    3가지 폴더(Huber, MAE, MSE)에서 B0047 잔차를 계산하고 z-score 표준화 후 평균을 구함
    """
    
    # 폴더 경로들
    folders = [
        'reconstructed_deepsc_Huber',
        'reconstructed_deepsc_MAE', 
        'reconstructed_deepsc_MSE'
    ]
    
    # 원본 데이터 경로 (B0047 원본 데이터)
    original_data_path = 'data_handling/merged/B0047.csv'

    model_names = ['Huber', 'MAE', 'MSE']
    
    # 원본 데이터 로드
    print("원본 데이터 로딩 중...")
    original_data = pd.read_csv(original_data_path)
    
    # 6개 피쳐 컬럼들 (전압, 전류, 온도, 충전용량, 방전용량, 시간)
    feature_columns = ['Voltage_measured', 'Current_measured', 'Temperature_measured', 
                      'Current_load', 'Voltage_load', 'Time']
    
    # 각 폴더의 잔차 데이터를 저장할 리스트
    all_residuals = []
    
    for folder in folders:
        reconstructed_file = f"{folder}/B0047_reconstructed.csv"
        
        if os.path.exists(reconstructed_file):
            print(f"{folder}에서 재구성 데이터 로딩 중...")
            reconstructed_data = pd.read_csv(reconstructed_file)
            
            # 잔차 계산 (원본 - 재구성)
            residuals = original_data[feature_columns] - reconstructed_data[feature_columns]
            
            # z-score 표준화
            residuals_zscore = pd.DataFrame()
            for col in feature_columns:
                residuals_zscore[col] = stats.zscore(residuals[col], nan_policy='omit')
            
            all_residuals.append(residuals_zscore)
            print(f"{folder} 잔차 계산 완료")
        else:
            print(f"경고: {reconstructed_file} 파일을 찾을 수 없습니다.")
    
    if not all_residuals:
        print("오류: 처리할 수 있는 잔차 데이터가 없습니다.")
        return
    
    # 3가지 방법의 잔차 평균 계산
    mean_residuals = pd.concat(all_residuals).groupby(level=0).mean()
    
    # 결과 저장
    output_file = save_dir+'B0047_residual_mean.csv'
    mean_residuals.to_csv(output_file, index=False)
    print(f"평균 잔차가 {output_file}에 저장되었습니다.")
    
    # 통계 정보 출력
    print("\n=== 잔차 통계 정보 ===")
    print(f"데이터 포인트 수: {len(mean_residuals)}")
    print("\n각 피쳐별 평균 잔차:")
    for col in feature_columns:
        print(f"{col}: {mean_residuals[col].mean():.6f}")
    
    print("\n각 피쳐별 잔차 표준편차:")
    for col in feature_columns:
        print(f"{col}: {mean_residuals[col].std():.6f}")
    
    # 시각화
    create_residual_plots(mean_residuals, feature_columns)
    plot_modelwise_residuals(all_residuals, feature_columns, model_names)

    return mean_residuals

def create_residual_plots(mean_residuals, feature_columns):
    """
    잔차 평균을 시각화하는 함수
    """
    
    # 1. 시간에 따른 잔차 변화
    plt.figure(figsize=(15, 10))
    
    for i, col in enumerate(feature_columns, 1):
        plt.subplot(2, 3, i)
        plt.plot(mean_residuals[col], alpha=0.7)
        plt.title(f'{col} - 잔차 변화')
        plt.xlabel('시간 인덱스')
        plt.ylabel('Z-score 표준화된 잔차')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir+'B0047_residual_mean_timeseries.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. 잔차 분포 히스토그램
    plt.figure(figsize=(15, 10))
    
    for i, col in enumerate(feature_columns, 1):
        plt.subplot(2, 3, i)
        plt.hist(mean_residuals[col], bins=50, alpha=0.7, edgecolor='black')
        plt.title(f'{col} - 잔차 분포')
        plt.xlabel('Z-score 표준화된 잔차')
        plt.ylabel('빈도')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}B0047_residual_mean_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. 상관관계 히트맵
    plt.figure(figsize=(10, 8))
    correlation_matrix = mean_residuals.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.3f')
    plt.title('피쳐별 잔차 상관관계')
    plt.tight_layout()
    plt.savefig(save_dir+'B0047_residual_mean_correlation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. 박스플롯
    plt.figure(figsize=(12, 6))
    mean_residuals.boxplot()
    plt.title('피쳐별 잔차 분포 (박스플롯)')
    plt.ylabel('Z-score 표준화된 잔차')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir+'B0047_residual_mean_boxplot.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_modelwise_residuals(all_residuals, feature_columns, model_names):
    """
    각 모델별 잔차(z-score) 시계열을 한 그래프에 겹쳐서 플롯
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(18, 10))
    for i, col in enumerate(feature_columns, 1):
        plt.subplot(2, 3, i)
        for residuals, name in zip(all_residuals, model_names):
            plt.plot(residuals[col], label=name, alpha=0.7)
        plt.title(f'{col} - 모델별 잔차 시계열')
        plt.xlabel('시간 인덱스')
        plt.ylabel('Z-score 표준화 잔차')
        plt.legend()
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir+'B0047_residuals_by_model_timeseries.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 히스토그램도 추가 (옵션)
    plt.figure(figsize=(18, 10))
    for i, col in enumerate(feature_columns, 1):
        plt.subplot(2, 3, i)
        for residuals, name in zip(all_residuals, model_names):
            plt.hist(residuals[col], bins=40, alpha=0.5, label=name, density=True)
        plt.title(f'{col} - 모델별 잔차 분포')
        plt.xlabel('Z-score 표준화 잔차')
        plt.ylabel('밀도')
        plt.legend()
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir+'B0047_residuals_by_model_hist.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # 잔차 평균 계산 및 시각화
    mean_residuals = calculate_residual_mean()
    
    if mean_residuals is not None:
        print("\n=== 분석 완료 ===")
        print("생성된 파일들:")
        print("- B0047_residual_mean.csv: 평균 잔차 데이터")
        print("- B0047_residual_mean_timeseries.png: 시간에 따른 잔차 변화")
        print("- B0047_residual_mean_distribution.png: 잔차 분포 히스토그램")
        print("- B0047_residual_mean_correlation.png: 피쳐별 상관관계")
        print("- B0047_residual_mean_boxplot.png: 박스플롯") 