import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from pathlib import Path

# Set matplotlib parameters
plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.unicode_minus"] = False

# Data paths
original_data_path = "/Users/yujinkang/time_series_DeepSC_adv/original_dataset/data"
metadata_path = "/Users/yujinkang/time_series_DeepSC_adv/original_dataset/metadata.csv"
new_data_path = "/Users/yujinkang/time_series_DeepSC_adv/current_measured_outlier/data_no_minus2.5_4"
# new_data_path = "/Users/yujinkang/time_series_DeepSC_adv/current_measured_outlier/data_no_minus2.5_4.5"


def remove_minus3_minus4_and_battery00049_00052():
    """Remove -3 group, -4 group data"""

    print("Starting -3 group, -4 group removal process...")

    source_data_path = original_data_path
    print("Using original data as source")

    csv_files = [f for f in os.listdir(source_data_path) if f.endswith(".csv")]
    csv_files.sort()

    print(f"Total files to process: {len(csv_files)}")

    # Statistics tracking
    total_original_rows = 0
    total_removed_minus3_rows = 0
    total_removed_minus4_rows = 0
    total_new_rows = 0
    files_processed = 0
    files_skipped = 0

    # Process each file
    for i, filename in enumerate(csv_files):
        if i % 500 == 0:
            print(f"Progress: {i}/{len(csv_files)}")

        try:
            source_file_path = os.path.join(source_data_path, filename)
            new_file_path = os.path.join(new_data_path, filename)

            # Read source data
            data = pd.read_csv(source_file_path)
            original_rows = len(data)
            total_original_rows += original_rows

            if "Current_measured" in data.columns:
                # Remove rows where Current_measured is in -3 group (-4.5 to -2.5)
                minus3_mask = (data["Current_measured"] >= -3.5) & (
                    data["Current_measured"] < -2.5
                )
                removed_minus3_rows = minus3_mask.sum()
                total_removed_minus3_rows += removed_minus3_rows

                # Remove rows where Current_measured is in -4 group (-4.5 to -3.5)
                minus4_mask = (data["Current_measured"] > -4) & (
                    data["Current_measured"] < -3.5
                )  # 2.5_4
                # minus4_mask = (data['Current_measured'] >= -4.5) & (data['Current_measured'] < -3.5)  # 2.5_4.5
                removed_minus4_rows = minus4_mask.sum()
                total_removed_minus4_rows += removed_minus4_rows

                # Keep rows not in -3 or -4 groups
                filtered_data = data[~(minus3_mask | minus4_mask)]
                new_rows = len(filtered_data)
                total_new_rows += new_rows

                # Save filtered data
                filtered_data.to_csv(new_file_path, index=False)
                files_processed += 1

                if removed_minus3_rows > 0 or removed_minus4_rows > 0:
                    print(
                        f"  {filename}: {original_rows} �� {new_rows} (removed -3: {removed_minus3_rows}, -4: {removed_minus4_rows})"
                    )
            else:
                # If no Current_measured column, copy as is
                data.to_csv(new_file_path, index=False)
                total_new_rows += original_rows
                files_processed += 1

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print(f"\n" + "=" * 70)
    print("MINUS 3 GROUP, MINUS 4 GROUP REMOVAL COMPLETED")
    print("=" * 70)
    print(f"Source data path: {source_data_path}")
    print(f"New data path: {new_data_path}")
    print(f"Files processed: {files_processed}")
    print(f"Total original rows: {total_original_rows:,}")
    print(f"Total removed rows (-3 group): {total_removed_minus3_rows:,}")
    print(f"Total removed rows (-4 group): {total_removed_minus4_rows:,}")
    print(
        f"Total removed rows (both groups): {total_removed_minus3_rows + total_removed_minus4_rows:,}"
    )
    print(f"Total new rows: {total_new_rows:,}")
    print(
        f"Removal percentage (-3 group): {total_removed_minus3_rows/total_original_rows*100:.2f}%"
    )
    print(
        f"Removal percentage (-4 group): {total_removed_minus4_rows/total_original_rows*100:.2f}%"
    )
    print(
        f"Total removal percentage: {(total_removed_minus3_rows + total_removed_minus4_rows)/total_original_rows*100:.2f}%"
    )
    print(f"Files removed percentage: {files_skipped/len(csv_files)*100:.2f}%")

    return new_data_path


def analyze_filtered_data(data_path):
    """Analyze the filtered data and create visualizations."""

    print(f"\nAnalyzing filtered data from: {data_path}")

    # Get all CSV files
    csv_files = [f for f in os.listdir(data_path) if f.endswith(".csv")]
    csv_files.sort()

    print(f"Files to analyze: {len(csv_files)}")

    # Collect Current_measured data
    all_current_data = []
    file_info = []

    print("Collecting data...")
    for i, filename in enumerate(csv_files):
        if i % 500 == 0:
            print(f"Progress: {i}/{len(csv_files)}")

        try:
            file_path = os.path.join(data_path, filename)
            data = pd.read_csv(file_path)

            if "Current_measured" in data.columns:
                current_values = data["Current_measured"].dropna()
                if len(current_values) > 0:
                    all_current_data.extend(current_values.tolist())
                    file_info.extend([filename] * len(current_values))
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print(f"Collected data count: {len(all_current_data):,}")

    # Create dataframe
    df = pd.DataFrame({"Current_measured": all_current_data, "filename": file_info})

    # Basic statistics
    print("\n=== Filtered Data Basic Statistics ===")
    print(df["Current_measured"].describe())

    # Outlier detection (IQR method)
    Q1 = df["Current_measured"].quantile(0.25)
    Q3 = df["Current_measured"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[
        (df["Current_measured"] < lower_bound) | (df["Current_measured"] > upper_bound)
    ]

    print(f"\n=== Outlier Statistics (IQR method) ===")
    print(f"Q1: {Q1:.6f}")
    print(f"Q3: {Q3:.6f}")
    print(f"IQR: {IQR:.6f}")
    print(f"Lower bound: {lower_bound:.6f}")
    print(f"Upper bound: {upper_bound:.6f}")
    print(f"Outlier count: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")

    # Z-score method
    z_scores = np.abs(
        (df["Current_measured"] - df["Current_measured"].mean())
        / df["Current_measured"].std()
    )
    zscore_outliers = df[z_scores > 3]

    print(f"\n=== Outlier Statistics (Z-score method, |z| > 3) ===")
    print(
        f"Z-score outlier count: {len(zscore_outliers)} ({len(zscore_outliers)/len(df)*100:.2f}%)"
    )

    # Create visualizations
    create_filtered_visualizations(df, outliers, lower_bound, upper_bound)

    return df, outliers


def create_filtered_visualizations(df, outliers, lower_bound, upper_bound):
    """Create visualizations for the filtered data analysis."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        "Current_measured Analysis(After Removing -2.5 Group, -4 Group)", fontsize=16
    )
    # fig.suptitle('Current_measured Analysis(After Removing -2.5 Group, -4.5 Group)', fontsize=16)

    # 1. Histogram (all data)
    axes[0, 0].hist(
        df["Current_measured"], bins=100, alpha=0.7, color="blue", edgecolor="black"
    )
    axes[0, 0].axvline(
        lower_bound, color="red", linestyle="--", label=f"Lower: {lower_bound:.6f}"
    )
    axes[0, 0].axvline(
        upper_bound, color="red", linestyle="--", label=f"Upper: {upper_bound:.6f}"
    )
    axes[0, 0].set_title("All Data Histogram)")
    axes[0, 0].set_xlabel("Current_measured")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Boxplot
    axes[0, 1].boxplot(df["Current_measured"])
    axes[0, 1].set_title("Boxplot")
    axes[0, 1].set_ylabel("Current_measured")
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Outlier histogram or data distribution
    if len(outliers) > 0:
        axes[0, 2].hist(
            outliers["Current_measured"],
            bins=50,
            alpha=0.7,
            color="red",
            edgecolor="black",
        )
        axes[0, 2].set_title("Outlier Histogram")
        axes[0, 2].set_xlabel("Current_measured")
        axes[0, 2].set_ylabel("Frequency")
    else:
        # If no outliers, show data distribution in different ranges
        axes[0, 2].hist(
            df["Current_measured"], bins=50, alpha=0.7, color="green", edgecolor="black"
        )
        axes[0, 2].set_title("Data Distribution(No Outliers Found)")
        axes[0, 2].set_xlabel("Current_measured")
        axes[0, 2].set_ylabel("Frequency")
    axes[0, 2].grid(True, alpha=0.3)

    # 4. Time series plot
    tdata = df.sample(len(df))
    axes[1, 0].scatter(range(len(tdata)), tdata["Current_measured"], alpha=0.6, s=1)
    axes[1, 0].axhline(lower_bound, color="red", linestyle="--", alpha=0.7)
    axes[1, 0].axhline(upper_bound, color="red", linestyle="--", alpha=0.7)
    axes[1, 0].set_title("Time Series Sample Plot")
    axes[1, 0].set_xlabel("Index")
    axes[1, 0].set_ylabel("Current_measured")
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Data distribution by file (top files with most data)
    file_counts = df["filename"].value_counts().head(20)
    axes[1, 1].bar(
        range(len(file_counts)),
        file_counts.values,
        alpha=0.7,
        color="blue",
        edgecolor="black",
    )
    axes[1, 1].set_title("Data Count by File")
    axes[1, 1].set_xlabel("File Index")
    axes[1, 1].set_ylabel("Data Count")
    axes[1, 1].grid(True, alpha=0.3)

    # 6. Data range comparison (before vs after filtering)
    # Create a comparison showing the data range
    data_ranges = df["Current_measured"].describe()
    axes[1, 2].bar(
        ["Min", "25%", "50%", "75%", "Max"],
        [
            data_ranges["min"],
            data_ranges["25%"],
            data_ranges["50%"],
            data_ranges["75%"],
            data_ranges["max"],
        ],
        alpha=0.7,
        color="purple",
        edgecolor="black",
    )
    axes[1, 2].set_title("Data Range Statistics\n(No -3 Group, No -4 Group)")
    axes[1, 2].set_ylabel("Current_measured Value")
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    # output_file = 'current_measured_outlier/plot/current_measured_no_minus2.5_4.png'
    # output_file = 'current_measured_outlier/plot/current_measured_no_minus2.5_4.5.png'
    # plt.savefig(output_file, dpi=300, bbox_inches='tight')
    # plt.show()

    # print(f"\nVisualization saved as '{output_file}'")


def main():
    """Main function to run the -3 group, -4 group removal and analysis."""

    print("Current_measured -3 Group, -4 Group Removal Tool")
    print("=" * 70)

    # Step 1: Remove -3 group, -4 group and battery_id B00049-B00052 files
    new_data_path = remove_minus3_minus4_and_battery00049_00052()

    # Step 2: Analyze filtered data and create visualizations
    df, outliers = analyze_filtered_data(new_data_path)

    print(f"\n" + "=" * 70)
    print("ANALYSIS COMPLETED!")
    # print(f"New data path: {new_data_path}")
    print(f"Visualization: current_measured_no_minus3_minus4_analysis.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
