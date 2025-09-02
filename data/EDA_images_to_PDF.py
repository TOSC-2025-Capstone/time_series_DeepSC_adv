import glob
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image


def create_battery_plots_pdf(base_directory, output_directory="battery_plots_pdfs"):
    """
    각 배터리 폴더의 plot 이미지들을 하나의 PDF로 정리하는 함수

    Args:
        base_directory (str): B0005, B0006 등 배터리 폴더들이 있는 상위 디렉토리
        output_directory (str): PDF 파일들을 저장할 디렉토리
    """

    # 피쳐 컬럼 정의
    feature_cols = [
        "Voltage_measured",
        "Current_measured",
        "Temperature_measured",
        "Current_load",
        "Voltage_load",
        # "Time",
    ]

    # 출력 디렉토리 생성
    os.makedirs(output_directory, exist_ok=True)

    # 배터리 폴더 찾기 (B0005 ~ B0056 패턴)
    battery_folders = []
    for folder in os.listdir(base_directory):
        if re.match(r"B\d{4}", folder):
            folder_path = os.path.join(base_directory, folder)
            if os.path.isdir(folder_path):
                battery_folders.append(folder_path)

    battery_folders.sort()  # 폴더명으로 정렬

    print(f"발견된 배터리 폴더 수: {len(battery_folders)}")

    # 각 배터리별로 PDF 생성
    for battery_folder in battery_folders:
        battery_id = os.path.basename(battery_folder)
        print(f"처리 중: {battery_id}")

        # PDF 파일 경로
        pdf_path = os.path.join(output_directory, f"{battery_id}_plots.pdf")

        # 이미지 파일들 찾기
        image_files = {}
        for feature in feature_cols:
            # 패턴: B0005_Voltage_measured.png 등
            pattern = os.path.join(battery_folder, f"{battery_id}_{feature}.png")
            matching_files = glob.glob(pattern)

            if matching_files:
                image_files[feature] = matching_files[0]
            else:
                print(f"  경고: {battery_id}_{feature}.png 파일을 찾을 수 없습니다.")

        # PDF 생성
        if image_files:
            create_single_battery_pdf(battery_id, image_files, pdf_path, feature_cols)
            print(f"  완료: {pdf_path}")
        else:
            print(f"  건너뜀: {battery_id} - 이미지 파일이 없습니다.")

    print(f"\n모든 PDF 파일이 '{output_directory}' 디렉토리에 생성되었습니다.")


def create_single_battery_pdf(battery_id, image_files, pdf_path, feature_cols):
    """
    하나의 배터리에 대한 PDF 파일 생성

    Args:
        battery_id (str): 배터리 ID (예: B0005)
        image_files (dict): 피쳐별 이미지 파일 경로
        pdf_path (str): 출력 PDF 파일 경로
        feature_cols (list): 피쳐 컬럼 리스트
    """

    with PdfPages(pdf_path) as pdf:
        # 2x3 그리드로 6개 플롯 배치
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(
            f"{battery_id} - All Features Analysis", fontsize=20, fontweight="bold"
        )

        axes = axes.flatten()  # 2D 배열을 1D로 변환

        for idx, feature in enumerate(feature_cols):
            if feature in image_files and os.path.exists(image_files[feature]):
                try:
                    # 이미지 로드 및 표시
                    img = Image.open(image_files[feature])
                    axes[idx].imshow(img)
                    axes[idx].set_title(f"{feature}", fontsize=14, fontweight="bold")
                    axes[idx].axis("off")  # 축 제거
                except Exception as e:
                    # 이미지 로드 실패 시 빈 플롯으로 표시
                    axes[idx].text(
                        0.5,
                        0.5,
                        f"{feature}\n(Image Load Failed)",
                        ha="center",
                        va="center",
                        fontsize=12,
                    )
                    axes[idx].set_title(f"{feature}", fontsize=14, fontweight="bold")
                    axes[idx].axis("off")
                    print(f"    이미지 로드 실패: {image_files[feature]} - {e}")
            else:
                # 파일이 없는 경우 빈 플롯으로 표시
                axes[idx].text(
                    0.5,
                    0.5,
                    f"{feature}\n(No Data)",
                    ha="center",
                    va="center",
                    fontsize=12,
                )
                axes[idx].set_title(f"{feature}", fontsize=14, fontweight="bold")
                axes[idx].axis("off")

        plt.tight_layout()
        pdf.savefig(fig, dpi=300, bbox_inches="tight")
        plt.close(fig)


def create_all_batteries_combined_pdf(
    base_directory, output_file="all_batteries_combined.pdf"
):
    """
    모든 배터리의 plot을 하나의 PDF에 정리 (선택적 기능)

    Args:
        base_directory (str): B0005, B0006 등 배터리 폴더들이 있는 상위 디렉토리
        output_file (str): 출력 PDF 파일명
    """

    feature_cols = [
        "Voltage_measured",
        "Current_measured",
        "Temperature_measured",
        "Current_load",
        "Voltage_load",
        # "Time",
    ]

    # 배터리 폴더 찾기
    battery_folders = []
    for folder in os.listdir(base_directory):
        if re.match(r"B\d{4}", folder):
            folder_path = os.path.join(base_directory, folder)
            if os.path.isdir(folder_path):
                battery_folders.append(folder_path)

    battery_folders.sort()

    print(f"통합 PDF 생성 중... ({len(battery_folders)}개 배터리)")

    with PdfPages(output_file) as pdf:
        for battery_folder in battery_folders:
            battery_id = os.path.basename(battery_folder)

            # 이미지 파일들 찾기
            image_files = {}
            for feature in feature_cols:
                pattern = os.path.join(battery_folder, f"{battery_id}_{feature}.png")
                matching_files = glob.glob(pattern)

                if matching_files:
                    image_files[feature] = matching_files[0]

            # 이 배터리의 페이지 생성
            if image_files:
                fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                fig.suptitle(
                    f"{battery_id} - All Features Analysis",
                    fontsize=20,
                    fontweight="bold",
                )

                axes = axes.flatten()

                for idx, feature in enumerate(feature_cols):
                    if feature in image_files and os.path.exists(image_files[feature]):
                        try:
                            img = Image.open(image_files[feature])
                            axes[idx].imshow(img)
                            axes[idx].set_title(
                                f"{feature}", fontsize=14, fontweight="bold"
                            )
                            axes[idx].axis("off")
                        except Exception as e:
                            axes[idx].text(
                                0.5,
                                0.5,
                                f"{feature}\n(Load Failed)",
                                ha="center",
                                va="center",
                                fontsize=12,
                            )
                            axes[idx].axis("off")
                    else:
                        axes[idx].text(
                            0.5,
                            0.5,
                            f"{feature}\n(No Data)",
                            ha="center",
                            va="center",
                            fontsize=12,
                        )
                        axes[idx].set_title(
                            f"{feature}", fontsize=14, fontweight="bold"
                        )
                        axes[idx].axis("off")

                plt.tight_layout()
                pdf.savefig(fig, dpi=300, bbox_inches="tight")
                plt.close(fig)

                print(f"  추가됨: {battery_id}")

    print(f"통합 PDF 완료: {output_file}")


def get_battery_summary(base_directory):
    """
    배터리 폴더와 파일 현황을 요약해서 보여주는 함수

    Args:
        base_directory (str): 배터리 폴더들이 있는 상위 디렉토리
    """

    feature_cols = [
        "Voltage_measured",
        "Current_measured",
        "Temperature_measured",
        "Current_load",
        "Voltage_load",
        # "Time",
    ]

    print("=== 배터리 데이터 현황 ===")

    battery_folders = []
    for folder in os.listdir(base_directory):
        if re.match(r"B\d{4}", folder):
            folder_path = os.path.join(base_directory, folder)
            if os.path.isdir(folder_path):
                battery_folders.append(folder_path)

    battery_folders.sort()

    print(f"총 배터리 폴더 수: {len(battery_folders)}")
    print(
        f"배터리 ID 범위: {os.path.basename(battery_folders[0])} ~ {os.path.basename(battery_folders[-1])}"
    )
    print()

    # 각 배터리별 파일 현황
    missing_files = []
    complete_batteries = []

    for battery_folder in battery_folders:
        battery_id = os.path.basename(battery_folder)
        missing_features = []

        for feature in feature_cols:
            pattern = os.path.join(battery_folder, f"{battery_id}_{feature}.png")
            if not glob.glob(pattern):
                missing_features.append(feature)

        if missing_features:
            missing_files.append((battery_id, missing_features))
        else:
            complete_batteries.append(battery_id)

    print(f"완전한 데이터를 가진 배터리: {len(complete_batteries)}개")
    print(f"누락된 파일이 있는 배터리: {len(missing_files)}개")

    if missing_files:
        print("\n--- 누락 파일 상세 ---")
        for battery_id, missing in missing_files:
            print(f"{battery_id}: {', '.join(missing)}")


# 사용 예시
if __name__ == "__main__":
    # 설정
    base_dir = "./EDA_images"  # 배터리 폴더들이 있는 디렉토리 경로로 변경하세요
    output_dir = "battery_plots_pdfs"

    # 1. 데이터 현황 확인
    get_battery_summary(base_dir)

    print("\n" + "=" * 50)

    # 2. 각 배터리별로 개별 PDF 생성
    # create_battery_plots_pdf(base_dir, output_dir)

    print("\n" + "=" * 50)

    # 3. 선택사항: 모든 배터리를 하나의 PDF로 통합
    create_all_batteries_combined_pdf(base_dir, "all_batteries_combined.pdf")

    print("\nPDF 생성 완료!")
