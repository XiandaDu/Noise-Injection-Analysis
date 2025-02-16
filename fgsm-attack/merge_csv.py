import os 
import pandas as pd

def concat_csvs(label_path, label):
    all_data = {
        'image': {f'e{i}': [] for i in range(10)},
        'by-depth': {f'e{i}': [] for i in range(10)}
    }

    count_image = 0
    count_by_depth = 0

    for image_folder in os.listdir(label_path):
        image_path = os.path.join(label_path, image_folder)
        if not os.path.isdir(image_path):
            continue

        # Determine the folder type
        if image_folder.endswith('-by-depth'):
            category = 'by-depth'
            count_by_depth += 1
            count = count_by_depth
        else:
            category = 'image'
            count_image += 1
            count = count_image

        # print("Processing", image_path)

        # Process CSV files for e0-e9
        for i in range(10):
            csv_file = f'original_features_{count}_e0.csv' if i == 0 else f'fgsm_features_{count}_e{i}.csv'
            csv_path = os.path.join(image_path, csv_file)

            # print("Trying to process", csv_file)
            if os.path.exists(csv_path):
                # print("Processing", csv_file)
                df = pd.read_csv(csv_path)
                df.insert(0, 'image', image_folder)  # Add image column
                df.insert(1, 'epsilon', f'e{i}')  # Add epsilon column
                all_data[category][f'e{i}'].append(df)

    # Merge and save CSV files separately for both folder types
    for category, data_dict in all_data.items():
        for key, data in data_dict.items():
            if data:
                final_df = pd.concat(data, ignore_index=True)
                suffix = '-by-depth' if category == 'by-depth' else ''
                output_file = os.path.join(label_path, f"big_{label}{suffix}_{key}.csv")
                final_df.to_csv(output_file, index=False)
                print(f"Saved {output_file}")

def main(root_dir):
    # Iterate through all label folders
    for label in os.listdir(root_dir):
        label_path = os.path.join(root_dir, label)
        if os.path.isdir(label_path):
            print(f"Processing label: {label}")
            concat_csvs(label_path, label)

if __name__ == "__main__":
    root_directory = "fgsm_results"  # Root directory
    main(root_directory)
