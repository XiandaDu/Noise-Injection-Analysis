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

def image_folder_statistics():
    base_dir = Path("./fgsm_results")
    
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
                if csv_filename.startswith("fgsm"):
                    # Assuming filename like "fgsm_e3.csv" (extract last 6-4 characters -> "e3")
                    label = csv_filename[-6:-4]
                elif csv_filename.startswith("original"):
                    label = "original"
                else:
                    label = "unknown"
                    continue

                compute_statistics(csv_file, str(folder_path), label)



def big_csv_statistics():
    base_dir = Path("./fgsm_results")
    
    for label_name in os.listdir(base_dir):
        label_path = base_dir / label_name
        
        # ---- 1. 先处理同目录下以 big_ 开头的 CSV 文件 ----
        big_csv_files = list(label_path.glob("big_*.csv"))
        for csv_file in big_csv_files:
            csv_filename = csv_file.name

            # 根据实际需求解析 label，这里和 fgsm 类似，只取末尾 e{数字}
            if csv_filename.startswith(f"big_{label_name}"):
                # 假设文件名形如 big_133_e3.csv 或 big_133-by-depth_e4.csv
                # 则可以按之前的方式提取最后两位作为 label
                if "by-depth" in csv_filename:
                    label = csv_filename[-15:-4]
                    print("matched ", csv_filename)
                else:
                    label = csv_filename[-6:-4]  # e0, e1, e2, ...
            else:
                label = "unknown"
                continue
            
            # 统计结果会保存在 label_path/statistics 里
            compute_statistics(csv_file, str(label_path), label)

def final_csv_statistics():
    base_dir = Path("./all-concat/final")
    
    # ---- 1. 先处理同目录下以 big_ 开头的 CSV 文件 ----
    final_csv_files = list(base_dir.glob("final*.csv"))

    for csv_file in final_csv_files:
        csv_filename = csv_file.name
        print(csv_filename)
        if "by-depth" in csv_filename:
            label = csv_filename[-15:-4]
            print("matched ", csv_filename)
        else:
            label = csv_filename[-6:-4]  # e0, e1, e2, ...
        
        compute_statistics(csv_file, str(base_dir), label)

if __name__ == "__main__":
    big_csv_statistics()
    final_csv_statistics()