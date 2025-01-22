import pandas as pd
import os
import matplotlib.pyplot as plt
import re

def compute_statistics_and_generate_histograms_by_layer(csv_path, output_dir, noise):
    statistics_dir = os.path.join(output_dir, "statistics-by-depth")
    os.makedirs(statistics_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    layer_columns = [col for col in df.columns if col.startswith("layer")]
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
        histogram_path = os.path.join(statistics_dir, f"{noise}_{col}_histogram.png")
        plt.savefig(histogram_path)
        plt.close()

    stats_df = pd.DataFrame(statistics).T
    stats_csv_path = os.path.join(statistics_dir, f"{noise}_layer_statistics.csv")
    stats_df.to_csv(stats_csv_path)
    print(f"Statistics and histograms saved in {statistics_dir}")

if __name__ == "__main__":
    image_name = "n01753488_177"
    csv_dir = f"./output-{image_name}-by-depth/aggregated"
    output_dir = f"./output-{image_name}-by-depth"

    for csv_file in os.listdir(csv_dir):
        if csv_file.endswith(".csv"):
            csv_path = os.path.join(csv_dir, csv_file)
            noise = csv_file.split('_')[-1].split('.')[0:2]
            noise = '.'.join(noise)
            compute_statistics_and_generate_histograms_by_layer(csv_path, output_dir, noise)
