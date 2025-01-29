import pandas as pd
import os
import matplotlib.pyplot as plt
from pathlib import Path

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
        histogram_path = os.path.join(statistics_dir, f"{label}_{col}_histogram.png")
        plt.savefig(histogram_path)
        plt.close()

    stats_df = pd.DataFrame(statistics).T
    stats_csv_path = os.path.join(statistics_dir, f"{label}_statistics.csv")
    stats_df.to_csv(stats_csv_path)
    print(f"Statistics and histograms saved in {statistics_dir}")



if __name__ == "__main__":
    for image_name in range(1,7):
        input_folder = Path(f"./fgsm_results/image{image_name}")

        for csv_file in input_folder.glob("*.csv"):
            if(str(csv_file).split('\\')[-1].startswith("fgsm")):
                label = str(csv_file)[-6:-4]
                output_dir = f"./fgsm_results/image{image_name}"
                compute_statistics_and_generate_histograms(csv_file, output_dir, label)
            elif(str(csv_file).split('\\')[-1].startswith("original")):
                label = "original"
                output_dir = f"./fgsm_results/image{image_name}"
                compute_statistics_and_generate_histograms(csv_file, output_dir, label)