import os 
import re
import pandas as pd
from pathlib import Path

def compute_statistics(csv_path, output_dir, label):
    """
    Read the CSV at csv_path and compute statistics for columns starting with 'Layer':
      - Calculate mean, median, std
      - Generate a CSV of statistics results
    """
    statistics_dir = os.path.join(output_dir, "statistics")
    os.makedirs(statistics_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    # Find all columns starting with "Layer"
    layer_columns = [col for col in df.columns if col.startswith("Layer")]

    statistics = {}
    for col in layer_columns:
        col_data = df[col]
        statistics[col] = {
            "mean": col_data.mean(),
            "median": col_data.median(),
            "std": col_data.std()
        }

    # Output statistics as CSV
    stats_df = pd.DataFrame(statistics).T
    stats_csv_path = os.path.join(statistics_dir, f"{label}_statistics.csv")
    stats_df.to_csv(stats_csv_path)
    print(f"Saved statistics CSV for {csv_path}: {stats_csv_path}")

if __name__ == "__main__":
    base_dir = Path("./gaussian_results")
    
    # Regex pattern: matches names like "image1" or "image1-by-depth"
    pattern = re.compile(r"^image\d+(-by-depth)?$")
    for label_name in os.listdir(base_dir):
        label_path = base_dir / label_name
        # Iterate through subdirectories, matching "image..." or "image...-by-depth"
        for folder_name in os.listdir(label_path):
            if not pattern.match(folder_name):
                continue  # Skip if it doesn't match the pattern
            
            folder_path = label_path / folder_name
            if not folder_path.is_dir():
                continue
            
            # Iterate through all CSV files in the folder
            for csv_file in folder_path.glob("*.csv"):
                csv_filename = csv_file.name

                # Determine label based on file prefix (modify logic as needed)
                if csv_filename.startswith("gaussian"):
                    # Assuming filename like "*_nl_0.3.csv" (extract last 10-4 characters -> "nl_0.3")
                    label = csv_filename[-10:-4]
                elif csv_filename.startswith("original"):
                    label = "original"
                else:
                    label = "unknown"
                    continue

                compute_statistics(csv_file, str(folder_path), label)
