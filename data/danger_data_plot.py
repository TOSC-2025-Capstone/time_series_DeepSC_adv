import pandas as pd
import matplotlib.pyplot as plt

# CSV 파일 읽기
df = pd.read_csv("./data/merged/B0005.csv")

# 필요한 컬럼 정의
features = [
    "Voltage_measured",
    "Current_measured",
    "Temperature_measured",
    "Current_load",
    "Voltage_load",
]

# cycle_idx로 그룹화
grouped = df.groupby("cycle_idx")

# 피처별 subplot 생성
fig, axes = plt.subplots(len(features), 1, figsize=(12, 3 * len(features)), sharex=True)

# 각 cycle에 대해 같은 plot에 선 추가
for cycle, group in grouped:
    for i, feature in enumerate(features):
        axes[i].plot(group["Time"], group[feature], label=f"Cycle {cycle}", alpha=0.6)

# subplot 타이틀 및 라벨 설정
for i, feature in enumerate(features):
    axes[i].set_ylabel(feature)
    axes[i].legend(loc="upper right", fontsize="small")
    axes[i].grid(True)

axes[-1].set_xlabel("Time")
plt.suptitle("All Cycles Overlaid by Feature", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.show()
