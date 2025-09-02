import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pdb

# --- 파라미터 ---
THRESHOLD = 0.05  # 기준 차이
TARGET_VALUES = [1, 2, 4]  # 확인할 기준값들

# 1. metadata.csv에서 discharge 파일 정보 추출
meta = pd.read_csv("original_dataset/metadata.csv")
discharge_data = meta[meta["type"] == "discharge"]
discharge_files = set(
    [f"{int(fname.split('.')[0]):05d}.csv" for fname in discharge_data["filename"]]
)
print(f"총 {len(discharge_files)}개의 discharge 파일 발견")

# 2. 각 파일 검사
folder = "original_dataset/data/"
file_range = range(1, 7566)  # 00001.csv ~ 07565.csv

problem_counts = {}  # 파일별 문제 row 개수 저장

for i in file_range:
    fname = f"{i:05d}.csv"
    # pdb.set_trace()
    if fname in discharge_files:
        # pdb.set_trace()
        fpath = os.path.join(folder, fname)
        if os.path.exists(fpath):
            # pdb.set_trace()
            df = pd.read_csv(fpath)

            if "Current_measured" in df.columns and "Current_load" in df.columns:
                # 절댓값 변환
                cm = df["Current_measured"].abs()
                cl = df["Current_load"].abs()

                # 기준값과의 최소 차이 계산
                cm_min_diff = cm.apply(lambda x: min(abs(x - v) for v in TARGET_VALUES))
                cl_min_diff = cl.apply(lambda x: min(abs(x - v) for v in TARGET_VALUES))

                # 기준 이상인 row 개수
                cm_problem = (cm_min_diff >= THRESHOLD).sum()
                cl_problem = (cl_min_diff >= THRESHOLD).sum()

                total_problem = cm_problem + cl_problem

                if total_problem > 0:
                    problem_counts[fname] = total_problem
                    print(
                        f"{fname}: 문제 {total_problem}개 (measured={cm_problem}, load={cl_problem})"
                    )

        else:
            print(f"{fname}: 파일 없음")

# 3. 통계 출력
if problem_counts:
    total_files = len(problem_counts)
    total_counts = sum(problem_counts.values())
    print("\n--- 통계 ---")
    print(f"문제 발생 파일 수: {total_files}")
    print(f"총 발생 개수: {total_counts}")
    print(f"파일당 평균 발생 개수: {total_counts/total_files:.2f}")

    # 시각화
    plt.figure(figsize=(12, 6))
    plt.hist(problem_counts.values(), bins=30, edgecolor="black")
    plt.title(
        f"Discharge 파일별 Current 값 기준 {TARGET_VALUES}에서 {THRESHOLD} 이상 차이 발생 분포"
    )
    plt.xlabel("문제 개수")
    plt.ylabel("파일 개수")
    plt.tight_layout()
    plt.savefig("discharge_current_problems_distribution.png")
    plt.show()
else:
    print("모든 파일에서 기준값 차이가 없습니다.")
