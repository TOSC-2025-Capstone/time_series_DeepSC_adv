import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. metadata.csv에서 discharge 파일 정보 추출
meta = pd.read_csv("original_dataset/metadata.csv")
discharge_data = meta[meta["type"] == "discharge"]
discharge_files = set(
    [f"{int(fname.split('.')[0]):05d}.csv" for fname in discharge_data["filename"]]
)
print(f"총 {len(discharge_files)}개의 discharge 파일 발견")

# 2. 각 파일의 row 수 세기
folder = "original_dataset/data/"
row_counts = []
file_range = range(1, 7566)  # 00001.csv ~ 07565.csv

for i in file_range:
    fname = f"{i:05d}.csv"
    # discharge 파일만 처리
    if fname in discharge_files:
        fpath = os.path.join(folder, fname)
        if os.path.exists(fpath):
            df = pd.read_csv(fpath)
            row_counts.append(len(df))
            print(f"{fname} (discharge): {len(df)} rows")
        else:
            print(f"{fname}: 파일 없음")

if row_counts:
    print("\n--- 통계 ---")
    print(f"파일 개수: {len(row_counts)}")
    print(f"평균 row 수: {sum(row_counts)/len(row_counts):.2f}")
    print(f"최소: {min(row_counts)}, 최대: {max(row_counts)}")

    # 시각화
    plt.figure(figsize=(12, 6))

    # 1. 히스토그램
    plt.subplot(1, 2, 1)
    plt.hist(row_counts, bins=30, edgecolor="black")
    plt.title("Discharge 파일 Row 수 분포")
    plt.xlabel("Row 수")
    plt.ylabel("파일 개수")

    # 2. Box plot
    plt.subplot(1, 2, 2)
    plt.boxplot(row_counts)
    plt.title("Row 수 Box Plot")
    plt.ylabel("Row 수")

    plt.tight_layout()
    plt.savefig("discharge_row_counts_distribution.png")
    plt.show()
else:
    print("파일이 없습니다.")
