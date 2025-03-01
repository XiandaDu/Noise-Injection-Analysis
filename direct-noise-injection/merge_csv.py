import os
import re
import pandas as pd

def extract_number(folder_name):
    """Extract number from folder name"""
    match = re.search(r'(\d+)', folder_name)
    return int(match.group(1)) if match else float('inf')  # Return infinity if no number is found

def concat_csvs(label_path, label):
    all_data = {
        'image': {f'sigma{i}': [] for i in range(10)},
        'by-depth': {f'sigma{i}': [] for i in range(10)}
    }

    # Read and sort folders by number
    image_folders = sorted(
        [f for f in os.listdir(label_path) if os.path.isdir(os.path.join(label_path, f))],
        key=extract_number
    )

    count_image = 0
    count_by_depth = 0

    for image_folder in image_folders:
        image_path = os.path.join(label_path, image_folder)

        # Determine category
        if image_folder.endswith('-by-depth'):
            category = 'by-depth'
            count_by_depth += 1
            count = count_by_depth
        else:
            category = 'image'
            count_image += 1
            count = count_image

        # Process CSV files for e0-e9
        for i in range(10):
            csv_file = f'original_features_{count}_nl_0.csv' if i == 0 else f'gaussian_features_{count}_nl_0.{i}.csv'
            csv_path = os.path.join(image_path, csv_file)

            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                df.insert(0, 'image', image_folder)  # Add image column
                df.insert(1, 'sigma', f'sigma{i}')  # Add epsilon column
                all_data[category][f'sigma{i}'].append(df)

    # Merge and save CSV files
    for category, data_dict in all_data.items():
        for key, data in data_dict.items():
            if data:
                final_df = pd.concat(data, ignore_index=True)
                suffix = '-by-depth' if category == 'by-depth' else ''
                output_file = os.path.join(label_path, f"big_{label}{suffix}_{key}.csv")
                final_df.to_csv(output_file, index=False)
                print(f"Saved {output_file}")

def main(root_dir):
    """Process all label directories"""
    for label in os.listdir(root_dir):
        label_path = os.path.join(root_dir, label)
        if os.path.isdir(label_path):
            print(f"Processing label: {label}")
            concat_csvs(label_path, label)

if __name__ == "__main__":
    root_directory = "gaussian_results"  # Root directory
    main(root_directory)
