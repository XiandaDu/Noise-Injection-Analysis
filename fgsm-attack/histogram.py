# import pandas as pd
# import os
# import matplotlib.pyplot as plt
# from pathlib import Path

# def compute_statistics_and_generate_histograms(csv_path, output_dir, label):
#     statistics_dir = os.path.join(output_dir, "statistics")
#     os.makedirs(statistics_dir, exist_ok=True)

#     df = pd.read_csv(csv_path)

#     layer_columns = [col for col in df.columns if col.startswith("Layer")]

#     statistics = {}

#     for col in layer_columns:
#         col_data = df[col]
        
#         statistics[col] = {
#             "mean": col_data.mean(),
#             "median": col_data.median(),
#             "std": col_data.std(),
#         }

#         plt.hist(col_data, bins=50, alpha=0.75)
#         plt.title(f"Histogram of {col}")
#         plt.xlabel("Value")
#         plt.ylabel("Frequency")
#         histogram_path = os.path.join(statistics_dir, f"{label}_{col}_histogram.png")
#         plt.savefig(histogram_path)
#         plt.close()

#     stats_df = pd.DataFrame(statistics).T
#     stats_csv_path = os.path.join(statistics_dir, f"{label}_statistics.csv")
#     stats_df.to_csv(stats_csv_path)
#     print(f"Statistics and histograms saved in {statistics_dir}")



# if __name__ == "__main__":
#     for image_name in range(1,7):
#         input_folder = Path(f"./fgsm_results/image{image_name}")

#         for csv_file in input_folder.glob("*.csv"):
#             if(str(csv_file).split('\\')[-1].startswith("fgsm")):
#                 label = str(csv_file)[-6:-4]
#                 output_dir = f"./fgsm_results/image{image_name}"
#                 compute_statistics_and_generate_histograms(csv_file, output_dir, label)
#             elif(str(csv_file).split('\\')[-1].startswith("original")):
#                 label = "original"
#                 output_dir = f"./fgsm_results/image{image_name}"
#                 compute_statistics_and_generate_histograms(csv_file, output_dir, label)


import os
import pandas as pd
import matplotlib.pyplot as plt

# Define input and output base directories
input_dir = "./fgsm_results"
output_base_dir = "histogram"
os.makedirs(output_base_dir, exist_ok=True)

# List of layers to process
layers = ["Layer_1", "Layer_2", "Layer_3", "Layer_4"]

# Loop over each label folder
for label in os.listdir(input_dir):
    print("Processing ", label)
    label_dir = os.path.join(input_dir, label)
    if not os.path.isdir(label_dir):
        continue
    # Create output directory for this label
    output_label_dir = os.path.join(output_base_dir, label)
    os.makedirs(output_label_dir, exist_ok=True)
    
    # Process each CSV file in the label folder
    for filename in os.listdir(label_dir):
        if not filename.endswith(".csv"):
            continue
        csv_path = os.path.join(label_dir, filename)
        
        # Read CSV with header so we can select columns by name
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Error reading {csv_path}: {e}")
            continue
        
        # Determine if file is by-depth or not based on filename
        if "by-depth" in filename:
            continue
        else:
            # Non by-depth file: columns are "Layer_1", "Layer_2", "Layer_3", "Layer_4"
            for layer in layers:
                if layer not in df.columns:
                    continue
                data = df[layer].values  # Select the column for the layer
                plt.figure()
                plt.hist(data, bins=50)  # English comment: Plot histogram with 50 bins
                plt.title(f"Histogram of {filename} - {layer}")
                
                out_filename = filename.replace(".csv", f"_{layer}_histogram.png")
                out_path = os.path.join(output_label_dir, out_filename)
                plt.savefig(out_path)
                plt.close()