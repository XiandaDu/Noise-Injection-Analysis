import pandas as pd
import os
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def compute_statistics_and_generate_histograms(csv_path, output_dir, label):
    statistics_dir = os.path.join(output_dir, "statistics")
    os.makedirs(statistics_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    layer_columns = [col for col in df.columns if col.startswith("Layer")]

    statistics = {}

    for col in layer_columns:
        col_data = df[col]
        
        statistics[col] = {
            "mean": col_data.mean(),
            "median": col_data.median(),
            "std": col_data.std(),
        }

        plt.hist(col_data, bins=50, alpha=0.75)
        plt.title(f"Histogram of {col}")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        histogram_path = os.path.join(statistics_dir, f"{label}_{col}_diff_histogram.png")
        plt.savefig(histogram_path)
        plt.close()

    stats_df = pd.DataFrame(statistics).T
    stats_csv_path = os.path.join(statistics_dir, f"{label}_diff_statistics.csv")
    stats_df.to_csv(stats_csv_path)
    print(f"Statistics and histograms saved in {statistics_dir}")


def diff(original_csv, modified_csv, output_path):
    df_org = pd.read_csv(original_csv)
    df_mod = pd.read_csv(modified_csv)
    
    if df_org.shape != df_mod.shape:
        print(f"Skipping {modified_csv} due to shape mismatch.")
        return
    
    diff_df = pd.DataFrame(np.subtract(df_org.values, df_mod.values), columns=df_org.columns)
    diff_df.to_csv(output_path, index=False)
    print(f"Saved difference CSV: {output_path}")


if __name__ == "__main__":
    for image_name in range(1, 7):
        input_folder = Path(f"./fgsm_results/image{image_name}")
        original_csv = None
        modified_csvs = []

        for csv_file in input_folder.glob("*.csv"):
            filename = csv_file.name
            if filename.startswith("original"):
                original_csv = csv_file
            elif filename.startswith("fgsm"):
                modified_csvs.append(csv_file)
        
        if original_csv:
            for mod_csv in modified_csvs:
                label = mod_csv.stem[-2:]  # e1, e2, e3, etc.
                statistics_dir = os.path.join(input_folder, "statistics-by-diff")
                os.makedirs(statistics_dir, exist_ok=True)
                output_path = statistics_dir+'/'+f"features-{image_name}-diff-org-{label}.csv"
                diff(original_csv, mod_csv, output_path)
        
        input_folder = Path(f"./fgsm_results/image{image_name}/statistics-by-diff")
        for csv_file in input_folder.glob("*.csv"):
            label = csv_file.stem[-6:] if csv_file.stem.startswith("features") else "original"
            compute_statistics_and_generate_histograms(csv_file, input_folder, label)
