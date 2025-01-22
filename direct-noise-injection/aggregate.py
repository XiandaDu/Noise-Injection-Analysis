import pandas as pd
import os

def aggregate_depths_and_save(csv_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for csv_file in os.listdir(csv_dir):
        if csv_file.endswith(".csv"):
            csv_path = os.path.join(csv_dir, csv_file)
            df = pd.read_csv(csv_path)

            layer_columns = [col for col in df.columns if col.startswith("layer")]
            aggregated_data = {}

            for col in layer_columns:
                layer_name = "_".join(col.split("_")[:1])
                if layer_name not in aggregated_data:
                    aggregated_data[layer_name] = df[col]
                else:
                    aggregated_data[layer_name] += df[col]

            aggregated_df = pd.DataFrame(aggregated_data)
            output_csv_path = os.path.join(output_dir, f"aggregated_{csv_file}")
            aggregated_df.to_csv(output_csv_path, index=False)
            print(f"Aggregated CSV saved to {output_csv_path}")

if __name__ == "__main__":
    image_name = "n01753488_177"
    csv_dir = f"./output-{image_name}-by-depth"
    output_dir = f"./output-{image_name}-by-depth/aggregated"
    aggregate_depths_and_save(csv_dir, output_dir)
