import pandas as pd
import os
import matplotlib.pyplot as plt

def compute_statistics_and_generate_histograms(csv_path, output_dir):
    statistics_dir = os.path.join(output_dir, "statistics")
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
        histogram_path = os.path.join(statistics_dir, f"{col}_histogram.png")
        plt.savefig(histogram_path)
        plt.close()

    stats_df = pd.DataFrame(statistics).T
    stats_csv_path = os.path.join(statistics_dir, "layer_statistics.csv")
    stats_df.to_csv(stats_csv_path)
    print(f"Statistics and histograms saved in {statistics_dir}")

if __name__ == "__main__":
    image_name = "n01753488_177"
    csv_path = f"./output-{image_name}/features_with_gaussian_noise.csv"
    output_dir = f"./output-{image_name}"
    compute_statistics_and_generate_histograms(csv_path, output_dir)