import os
import re
import pandas as pd

def compute_differences(input_dir, output_dir):
    """
    Compute element-wise differences between CSV files for different epsilon values.
    For each type (non by-depth and by-depth), the difference is computed as:
      DataFrame(e{i}) - DataFrame(e{j}), for each pair with i < j.
    Only numeric feature columns are used (dropping 'image' and 'epsilon').
    """
    # Dictionary to store file paths for each type and epsilon value.
    files_dict = {
        'non_depth': {},  # Files with names: final_e{i}.csv
        'by_depth': {}    # Files with names: final_by-depth_e{i}.csv
    }
    
    # Iterate over all CSV files in the input directory.
    for filename in os.listdir(input_dir):
        if not filename.endswith('.csv'):
            continue
        filepath = os.path.join(input_dir, filename)
        # Determine file type based on filename prefix.
        if filename.startswith("final_by-depth_"):
            match = re.search(r'final_by-depth_e(\d+)\.csv', filename)
            if match:
                eps = int(match.group(1))
                files_dict['by_depth'][eps] = filepath
        elif filename.startswith("final_"):
            match = re.search(r'final_e(\d+)\.csv', filename)
            if match:
                eps = int(match.group(1))
                files_dict['non_depth'][eps] = filepath

    # Ensure the output directory exists.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Compute differences for non by-depth files.
    for i in sorted(files_dict['non_depth'].keys()):
        for j in sorted(files_dict['non_depth'].keys()):
            if i < j:
                # Load CSV files for epsilon i and epsilon j.
                df_i = pd.read_csv(files_dict['non_depth'][i])
                df_j = pd.read_csv(files_dict['non_depth'][j])
                # Drop identifier columns (assumed to be 'image' and 'epsilon').
                df_i_feat = df_i.drop(columns=["image", "epsilon"], errors='ignore')
                df_j_feat = df_j.drop(columns=["image", "epsilon"], errors='ignore')
                # Compute element-wise difference (e{i} - e{j}).
                diff_df = df_i_feat - df_j_feat
                # Construct output filename.
                out_filename = f"big_e{i}_by_e{j}.csv"
                out_filepath = os.path.join(output_dir, out_filename)
                diff_df.to_csv(out_filepath, index=False)
                print(f"Saved {out_filepath}")

    # Compute differences for by-depth files.
    for i in sorted(files_dict['by_depth'].keys()):
        for j in sorted(files_dict['by_depth'].keys()):
            if i < j:
                df_i = pd.read_csv(files_dict['by_depth'][i])
                df_j = pd.read_csv(files_dict['by_depth'][j])
                df_i_feat = df_i.drop(columns=["image", "epsilon"], errors='ignore')
                df_j_feat = df_j.drop(columns=["image", "epsilon"], errors='ignore')
                diff_df = df_i_feat - df_j_feat
                out_filename = f"big_by-depth_e{i}_by_e{j}.csv"
                out_filepath = os.path.join(output_dir, out_filename)
                diff_df.to_csv(out_filepath, index=False)
                print(f"Saved {out_filepath}")

if __name__ == "__main__":
    input_directory = "./all-concat"  # Directory containing merged CSV files
    output_directory = "./diff"       # Directory to save the difference CSV files
    compute_differences(input_directory, output_directory)
