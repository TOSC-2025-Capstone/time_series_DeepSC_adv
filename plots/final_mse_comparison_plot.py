import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# 1. 데이터 정의 및 구조화
# 주어진 데이터를 pandas DataFrame 형태로 정리합니다.
data = {
    "SNR": [
        5,
        5,
        5,
        5,
        10,
        10,
        10,
        10,
        15,
        15,
        15,
        15,
        20,
        20,
        20,
        20
    ],
    "Model": [
        "LSTM",
        "GRU",
        "Transformer",
        "Inverted-Transformer",
        "LSTM",
        "GRU",
        "Transformer",
        "Inverted-Transformer",
        "LSTM",
        "GRU",
        "Transformer",
        "Inverted-Transformer",
        "LSTM",
        "GRU",
        "Transformer",
        "Inverted-Transformer"
    ],
    "MSE": [
        2.938732, # 5dB
        6.635599,
        2.374086,
        1.88575,
        0.98031,  # 10dB
        2.153676,
        0.638626,
        0.665384,
        0.391907, # 15dB
        0.699424,
        0.319927,
        0.362732,
        0.217569, # 20dB
        0.258287,
        0.252607,
        0.279395
    ]
}

df = pd.DataFrame(data)

# 2. 논문용 회색조 스타일 및 마커/선 스타일 정의
models = df['Model'].unique()
num_models = len(models)

# Greys 컬러맵을 사용하여 어두운 계열의 색상을 선택 (흰색은 피함)
# vmin/vmax를 설정하여 너무 밝거나 너무 어두운 색상 범위를 조정합니다.
cmap = cm.get_cmap('Greys')
# 0.3 (밝은 회색)부터 0.8 (진한 회색) 사이의 색상 4개를 추출
grayscale_colors = [cmap(i) for i in np.linspace(0.3, 0.8, num_models)]

# 논문 가독성을 높이기 위한 마커와 선 스타일 조합
markers = ['o', 's', '^', 'D'] # 원, 사각형, 삼각형, 마름모
linestyles = ['-', '--', ':', '-.'] # 실선, 점선, 아주 짧은 점선, 파선-점선

style_map = {
    model: {
        'color': grayscale_colors[i],
        'marker': markers[i],
        'linestyle': linestyles[i]
    }
    for i, model in enumerate(models)
}

# 3. 꺾은선 그래프 생성 및 시각화
plt.figure(figsize=(10, 6))

for model in models:
    subset = df[df['Model'] == model]
    style = style_map[model]

    plt.plot(
        subset['SNR'],
        subset['MSE'],
        color=style['color'],
        marker=style['marker'],
        linestyle=style['linestyle'],
        label=model,
        linewidth=2.0, # 선 두께 증가 (가독성 향상)
        markersize=7.0 # 마커 크기 증가
    )

# 그래프 제목 및 축 레이블 설정
plt.title('Model Performance Comparison (MSE vs. SNR)', fontsize=16)
plt.xlabel('SNR (dB)', fontsize=14)
plt.ylabel('Average MSE (no Time) [Log Scale]', fontsize=14)

# x축 눈금을 명시된 SNR 값만 표시하도록 설정
plt.xticks(df['SNR'].unique())

# 범례 표시 및 격자 추가
# 범례 프레임을 제거하고 배경을 흰색으로 설정하여 깔끔하게 보입니다.
plt.legend(title='Model', loc='upper right', frameon=False)
plt.grid(True, linestyle='-', alpha=0.3)

# y축 로그 스케일 적용 (MSE 변화 폭이 큼)
plt.yscale('log')

# 테두리를 깨끗하게 정리
plt.tight_layout()

# 그래프 저장 (논문용 고해상도 PNG 또는 PDF 권장)
plt.savefig('case43_mse_comparison_grayscale.png', dpi=300)
print("Graph saved as 'case43_mse_comparison_grayscale.png' with grayscale contrast.")

# 결과 DataFrame 출력 (확인용)
print("\nDataFrame for Plotting:")
print(df)
