### README
⚠️This repository is under active development, so the README.md may not always be up to date.

# Feature Map Analysis for ResNet Layers

## Overview

This repository is a utility for analyzing and visualizing the **feature maps** extracted from various layers of a ResNet model. The repository focuses on generating, aggregating, and visualizing feature maps to understand how different layers process data under various conditions, such as Gaussian noise injection. The outputs include statistical insights and visualizations of feature maps.

---

## Key Features

1. **Feature Extraction and Noise Injection**:
   - Extracts feature maps from different layers (`layer1`, `layer2`, `layer3`, `layer4`) of a ResNet model.
   - Injects different levels of Gaussian noise into the input image.
   - Saves the feature maps for each depth of each layer in CSV format.

2. **Aggregation by Depth**:
   - Aggregates the feature maps across depths for each layer into a single CSV file.

3. **Statistical Analysis**:
   - Computes statistics such as `mean`, `median`, and `standard deviation` for feature maps.
   - Generates histograms for feature distributions by depth and layer.

4. **Visualization of Feature Maps**:
   - Creates grayscale images for feature maps.
   - For each layer:
     - Groups the depths into batches of 64.
     - Generates images with a resolution specific to each layer:
       - `layer1`: `56x56`
       - `layer2`: `28x28`
       - `layer3`: `14x14`
       - `layer4`: `7x7`
     - Combines 64 feature maps into an `8x8` grid for visualization.

---

## Folder Structure

```
repository/
├── aggregated/                    # Aggregated CSV files by layer
├── output-n01753488_177-by-depth/ # Per-depth CSV files for each layer
├── output-n01753488_177-feature-maps/ # Grayscale images of feature maps
├── inject-gaussian-noise.py       # Script for noise injection and feature extraction
├── diagram.py                     # Script for visualizing feature maps
├── histogram-by-depth.py          # Script for depth-wise statistical analysis
├── imagenet_classes.txt           # List of ImageNet class labels
└── LICENSE                        # License file
```

---

## Workflow

1. **Feature Extraction**:
   - The `inject-gaussian-noise.py` script processes an input image through a ResNet model.
   - Feature maps are extracted for `layer1`, `layer2`, `layer3`, and `layer4`.
   - Noise levels ranging from `0.2` to `1.0` are applied, and the results are saved in CSV files.

2. **Aggregation**:
   - The `histogram-by-depth.py` script aggregates feature maps by summing across depths for each layer.
   - Aggregated results are stored in a new set of CSV files.

3. **Visualization**:
   - The `diagram.py` script visualizes feature maps as grayscale images.
   - Each 64 feature maps are grouped and visualized on a grid of `8x8` subplots, with layer-specific resolutions.

4. **Statistical Analysis**:
   - Generates statistical summaries (mean, median, std) and histograms for feature maps, either by depth or by layer.

---

## Outputs

- **CSV Files**:
  - `features_original.csv`: Feature maps without noise.
  - `features_noised_X.csv`: Feature maps with Gaussian noise of level `X`.

- **Visualizations**:
  - Grayscale images of feature maps for each layer, grouped into depths of 64.
  - Histograms representing the value distribution of feature maps.

---

## Example Usage

1. Extract feature maps and inject noise:
   ```bash
   python inject-gaussian-noise.py
   ```

2. Aggregate feature maps by depth:
   ```bash
   python histogram-by-depth.py
   ```

3. Visualize feature maps as grayscale images:
   ```bash
   python diagram.py
   ```

---

## Requirements

- Python 3.8+
- Dependencies:
  - `torch`
  - `torchvision`
  - `pandas`
  - `matplotlib`
  - `Pillow`

Install dependencies using:
```bash
pip install torch torchvision pandas matplotlib Pillow
```

---

## Author

This repository was created to facilitate deep learning feature map analysis for research purposes. For any questions or suggestions, feel free to contribute or contact the author.
