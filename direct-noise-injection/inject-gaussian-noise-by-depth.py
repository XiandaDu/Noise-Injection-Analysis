import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm

def save_features_to_csv_by_depth(features_dict, file_name):
    """
    Save network layer features to a CSV file in a by-depth format.
    Example structure of features_dict (where value is a [1, C, H, W] Tensor):
    {
        "layer1": <Tensor shape=[1, C1, H1, W1]>,
        "layer2": <Tensor shape=[1, C2, H2, W2]>,
        ...
    }
    Each channel (depth) of each layer will be flattened into a column.
    """
    # Map model layer names to desired column name prefixes
    layer_map = {
        "layer1": "Layer_1",
        "layer2": "Layer_2",
        "layer3": "Layer_3",
        "layer4": "Layer_4"
    }
    
    data = {}
    max_length = 0
    
    # Iterate through all layers
    for layer_name, layer_data in features_dict.items():
        _, channels, _, _ = layer_data.shape
        prefix_name = layer_map.get(layer_name, layer_name)
        
        # Flatten each channel
        for depth in range(channels):
            feature_values = layer_data[0, depth].contiguous().view(-1).cpu().numpy()
            max_length = max(max_length, len(feature_values))
            column_name = f"{prefix_name}_depth{depth}"
            data[column_name] = feature_values
    
    # Align lengths (some layers may have different flattened lengths)
    for key, values in data.items():
        if len(values) < max_length:
            data[key] = list(values) + [float("nan")] * (max_length - len(values))
    
    # Write to CSV
    df = pd.DataFrame(data)
    df.to_csv(file_name, index=False)
    print(f"Saved features CSV: {file_name}")

def inject_noise(image_tensor, noise_level):
    """
    Add random Gaussian noise to the tensor.
    """
    noise = torch.randn_like(image_tensor) * noise_level
    return image_tensor + noise

def main():
    # 1. Define and load the model
    model = models.resnet18(pretrained=True)
    model.eval()
    
    # Hook to get intermediate features
    features = {}
    def hook(module, input, output):
        features[module] = output

    layers_to_hook = ["layer1", "layer2", "layer3", "layer4"]
    for name, layer in model.named_modules():
        if name in layers_to_hook:
            layer.register_forward_hook(hook)

    # 2. Define image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    # 3. Dataset root, output directory, and noise levels
    dataset_root = "../Domain-Specific-Dataset/val"
    output_root = "./gaussian_results"
    os.makedirs(output_root, exist_ok=True)
    
    noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    # 4. Iterate through label folders
    for label in os.listdir(dataset_root):
        label_path = os.path.join(dataset_root, label)
        if not os.path.isdir(label_path):
            continue
        
        print(f"\nProcessing label: {label}")
        output_label_dir = os.path.join(output_root, label)
        os.makedirs(output_label_dir, exist_ok=True)
        
        # Read a subset of images as examples
        image_files = [f for f in os.listdir(label_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:40]
        
        # 5. Iterate through each image
        for idx, image_file in enumerate(tqdm(image_files, desc=f"Processing images for {label}")):
            image_path = os.path.join(label_path, image_file)
            image_output_dir = os.path.join(output_label_dir, f"image{idx+1}-by-depth")
            os.makedirs(image_output_dir, exist_ok=True)
            
            # Load and preprocess the image
            image = Image.open(image_path).convert("RGB")
            input_tensor = transform(image).unsqueeze(0)
            
            # Forward pass for the original image to extract features
            features.clear()
            with torch.no_grad():
                _ = model(input_tensor)
            
            # Extract original image features (by-depth)
            original_features = {
                name: features[layer]
                for name, layer in model.named_modules()
                if name in layers_to_hook and layer in features
            }
            
            # Save original features
            csv_orig_feat = os.path.join(image_output_dir, f"original_features_{idx+1}_nl_0.csv")
            save_features_to_csv_by_depth(original_features, csv_orig_feat)
            
            # 6. Iterate through each noise level and save features
            for nl in noise_levels:
                # Add noise and forward pass
                noisy_input = inject_noise(input_tensor, nl)
                
                features.clear()
                with torch.no_grad():
                    _ = model(noisy_input)
                
                # Extract noisy image features (by-depth)
                noisy_features = {
                    name: features[layer]
                    for name, layer in model.named_modules()
                    if name in layers_to_hook and layer in features
                }
                
                # Save noisy features
                csv_noisy_feat = os.path.join(image_output_dir, f"gaussian_features_{idx+1}_nl_{nl}.csv")
                save_features_to_csv_by_depth(noisy_features, csv_noisy_feat)

if __name__ == "__main__":
    main()
