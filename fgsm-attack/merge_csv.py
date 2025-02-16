import os
import pandas as pd

def concat_csvs(label_path, label):
    label_name = os.path.basename(label_path)
    all_data = {f'e{i}': [] for i in range(10)}  # store the data from e0-e9

    # Iterate through every image-? folder
    count = 0
    for image_folder in os.listdir(label_path):
        image_path = os.path.join(label_path, image_folder)
        if not os.path.isdir(image_path):
            continue
        count += 1
        # Find csv files from e0-e9
        for i in range(10):
            csv_file = f'original_features_{count}_e0.csv' if i == 0 else f'fgsm_features_{count}_e{i}.csv'
            csv_path = os.path.join(image_path, csv_file)
            
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                df.insert(0, 'image', image_folder)  # Add image label
                df.insert(1, 'epsilon', f'e{i}')  # Add epsilon value
                all_data[f'e{i}'].append(df)
    
    # Merge the data and save
    for key, data in all_data.items():
        if data:
            final_df = pd.concat(data, ignore_index=True)
            final_df.to_csv(os.path.join(label_path, f"big_{label}_{key}.csv"), index=False)
            print(f"Saved {label_name}/big_{label}_{key}.csv")

def main(root_dir):
    # Iterate through every label
    for label in os.listdir(root_dir):
        label_path = os.path.join(root_dir, label)
        if os.path.isdir(label_path):
            print(label)
            concat_csvs(label_path, label)

if __name__ == "__main__":
    root_directory = "fgsm_results"  # Root folder which contains fgsm_results
    main(root_directory)
