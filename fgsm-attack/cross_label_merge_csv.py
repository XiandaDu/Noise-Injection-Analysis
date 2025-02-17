import os
import re
import pandas as pd

def extract_number(folder_name):
    """Extract numeric part from folder name for sorting."""
    match = re.search(r'(\d+)', folder_name)
    return int(match.group(1)) if match else float('inf')

def concat_all_labels(root_dir, output_dir, i):
    # Global dictionary: category -> epsilon -> list of DataFrames
    global_data = {
        'image': {f'e{i}': [] for i in range(10)},
        'by-depth': {f'e{i}': [] for i in range(10)}
    }
    
    # Iterate over all label directories in the root directory
    for label in os.listdir(root_dir):
        label_path = os.path.join(root_dir, label)
        if not os.path.isdir(label_path):
            continue
        
        # Get all subfolders in the current label directory and sort them numerically
        subfolders = sorted(
            [f for f in os.listdir(label_path) if os.path.isdir(os.path.join(label_path, f))],
            key=extract_number
        )
        
        count_image = 0
        count_by_depth = 0
        
        # Process each subfolder (representing one image)
        for folder in subfolders:
            folder_path = os.path.join(label_path, folder)
            # Determine category based on folder name suffix
            if folder.endswith('-by-depth'):
                category = 'by-depth'
                count_by_depth += 1
                count = count_by_depth
            else:
                category = 'image'
                count_image += 1
                count = count_image
                
            # Process CSV files for each epsilon e0-e9
            if i == 0:
                csv_file = f'original_features_{count}_e0.csv'
            else:
                csv_file = f'fgsm_features_{count}_e{i}.csv'
            csv_path = os.path.join(folder_path, csv_file)
            
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                except Exception as e:
                    print(f"Error reading {csv_path}: {e}")
                    continue
                
                # Insert identifier columns if not present
                if 'image' not in df.columns:
                    df.insert(0, 'image', folder)
                if 'epsilon' not in df.columns:
                    df.insert(1, 'epsilon', f'e{i}')
                
                # Append the entire DataFrame for this epsilon and category
                global_data[category][f'e{i}'].append(df)
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Concatenate and save CSV files for each category and each epsilon
    for category, eps_dict in global_data.items():
        for epsilon, df_list in eps_dict.items():
            if df_list:
                final_df = pd.concat(df_list, ignore_index=True)
                # Suffix for by-depth category
                suffix = "-by-depth" if category == 'by-depth' else ""
                output_filename = f"final{suffix}_{epsilon}.csv"
                output_path = os.path.join(output_dir, output_filename)
                final_df.to_csv(output_path, index=False)
                print(f"Saved {output_path}")

if __name__ == "__main__":
    root_directory = "fgsm_results"    # Root directory containing all label folders
    output_directory = "./all-concat"    # Output directory for merged CSV files
    for i in range(10):
        concat_all_labels(root_directory, output_directory, i)
