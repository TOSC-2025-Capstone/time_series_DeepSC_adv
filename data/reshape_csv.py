import os
import pandas as pd

# 메타데이터 로딩
meta_df = pd.read_csv("discharge_metadata_with_cycle_idx.csv")

# 데이터 폴더 경로 (현재 디렉토리 기준)
data_dir = "./data"

# 결과 저장 폴더 (없으면 생성)
output_dir = "./merged"
os.makedirs(output_dir, exist_ok=True)

# 배터리 ID별로 그룹핑
for battery_id, group in meta_df.groupby("battery_id"):
    merged_df = pd.DataFrame()

    for _, row in group.iterrows():
        filename = row['filename']
        cycle_idx = row['cycle_idx']

        file_path = os.path.join(data_dir, filename)

        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                df['cycle_idx'] = cycle_idx  # 각 파일에 cycle_idx 추가
                merged_df = pd.concat([merged_df, df], ignore_index=True)
            except Exception as e:
                print(f"파일 {filename} 로딩 실패: {e}")
        else:
            print(f"파일 없음: {filename}")

    # 통합된 결과를 battery_id.csv로 저장
    output_path = os.path.join(output_dir, f"{battery_id}.csv")
    merged_df.to_csv(output_path, index=False)
    print(f"저장 완료: {output_path}")
