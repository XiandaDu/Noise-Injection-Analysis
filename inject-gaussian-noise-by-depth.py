import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import os

def process_gaussian_noise(image_path, output_dir, imagenet_classes_path):
    model = models.resnet18(weights=True)
    model.eval()

    features = {}

    def hook(module, input, output):
        features[module] = output

    layers_to_hook = ["layer1", "layer2", "layer3", "layer4"]
    for name, layer in model.named_modules():
        if name in layers_to_hook:
            layer.register_forward_hook(hook)

    with open(imagenet_classes_path) as f:
        imagenet_classes = [line.strip() for line in f.readlines()]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    def inject_noise(image_tensor, noise_level):
        noise = torch.randn_like(image_tensor) * noise_level
        return image_tensor + noise

    def save_image_from_tensor(image_tensor, file_name):
        unnormalize = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        )
        unnormalized_tensor = unnormalize(image_tensor.squeeze(0))
        clipped_tensor = torch.clamp(unnormalized_tensor, 0, 1)
        noisy_image = transforms.ToPILImage()(clipped_tensor)
        noisy_image.save(file_name)

    def save_features_to_csv_by_depth(features_dict, file_name):
        data = {}
        max_length = 0

        for layer_name, layer_data in features_dict.items():
            for depth in range(layer_data.shape[1]):
                feature_values = layer_data[0, depth].view(-1).numpy()
                max_length = max(max_length, len(feature_values))
                data[f"{layer_name}_depth{depth}"] = feature_values

        for key, values in data.items():
            if len(values) < max_length:
                data[key] = list(values) + [float("nan")] * (max_length - len(values))

        df = pd.DataFrame(data)
        df.to_csv(file_name, index=False)
        print(f"Features saved to {file_name}")


    noise_levels = [0.2, 0.4, 0.6, 0.8, 1.0]
    features_dict = {}

    # Extract original features
    with torch.no_grad():
        model(input_tensor)
    original_features = {name: features[layer] for name, layer in model.named_modules() if name in layers_to_hook}
    features_dict["original"] = original_features

    for noise_level in noise_levels:
        noisy_input = inject_noise(input_tensor, noise_level)
        noisy_image_path = os.path.join(output_dir, f"noisy_image_{noise_level}.jpeg")
        save_image_from_tensor(noisy_input, noisy_image_path)

        features.clear()
        with torch.no_grad():
            model(noisy_input)

        noisy_features = {name: features[layer] for name, layer in model.named_modules() if name in layers_to_hook}
        features_dict[f"noised_{noise_level}"] = noisy_features

    # Save original and noisy features by depth to CSV
    for noise_level in ["original"] + [f"noised_{level}" for level in noise_levels]:
        csv_path = os.path.join(output_dir, f"features_{noise_level}.csv")
        save_features_to_csv_by_depth(features_dict[noise_level], csv_path)

    print(f"Features and noisy images saved in {output_dir}")

if __name__ == "__main__":
    image_name = "n01753488_177"
    image_path = f"./Training/{image_name}.JPEG"
    output_dir = f"./output-{image_name}-by-depth"
    imagenet_classes_path = "imagenet_classes.txt"
    os.makedirs(output_dir, exist_ok=True)
    process_gaussian_noise(image_path, output_dir, imagenet_classes_path)
