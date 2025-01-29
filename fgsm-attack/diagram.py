import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def save_feature_map_as_image(feature_map, grid_dim, image_dim, output_path):
    """
    Combines smaller feature maps into a grid and saves as an image.

    Parameters:
        feature_map: np.array
            Array of shape (n_images, image_height, image_width).
        grid_dim: tuple
            Dimensions of the grid (rows, cols).
        image_dim: tuple
            Dimensions of each small image (height, width).
        output_path: str
            Path to save the generated image.
    """
    fig, axes = plt.subplots(*grid_dim, figsize=(10, 10))
    axes = axes.flatten()

    for idx, ax in enumerate(axes):
        if idx < len(feature_map):
            ax.imshow(feature_map[idx], cmap="gray")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def process_csv_files(input_folder, output_folder):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)

    grid_settings = {
        "Layer_1": (64, (56, 56)),
        "Layer_2": (128, (28, 28)),
        "Layer_3": (256, (14, 14)),
        "Layer_4": (512, (7, 7)),
    }

    for csv_file in input_folder.glob("*.csv"):
        df = pd.read_csv(csv_file)
        csv_folder = output_folder / csv_file.stem
        csv_folder.mkdir(parents=True, exist_ok=True)

        for layer_name, (total_depth, img_dim) in grid_settings.items():
            layer_columns = [col for col in df.columns if col.startswith(layer_name)]

            if len(layer_columns) != total_depth:
                print(f"Warning: {layer_name} in {csv_file.name} has {len(layer_columns)} columns, "
                      f"but expected {total_depth} columns.")
                continue

            for i in range(0, total_depth, 64):
                sub_columns = layer_columns[i:i + 64]
                sub_data = df[sub_columns].dropna().values
                feature_maps = sub_data.T.reshape((64, *img_dim))

                output_path = csv_folder / f"{layer_name}_part{i // 64 + 1}.png"
                save_feature_map_as_image(
                    feature_maps, grid_dim=(8, 8), image_dim=img_dim, output_path=output_path
                )
        print("Finished One CSV")


if __name__ == "__main__":
    for image_name in range(1,7):
        input_folder = Path(f"./fgsm_results/image{image_name}-by-depth")
        output_folder = input_folder / "diagram"
        process_csv_files(input_folder, output_folder)
