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

    def save_features_to_csv(features_dict, file_name):
        columns = []
        for noise_level, data in features_dict.items():
            for key, values in data.items():
                columns.append(pd.Series(values, name=f"{key}_{noise_level}"))
        df = pd.concat(columns, axis=1)
        df.to_csv(file_name, index=False)

    noise_levels = [0.2, 0.4, 0.6, 0.8, 1.0]
    features_dict = {}

    with torch.no_grad():
        model(input_tensor)
    original_features = {name: features[layer].view(-1).numpy() for name, layer in model.named_modules() if name in layers_to_hook}

    features_dict["original"] = original_features

    for noise_level in noise_levels:
        noisy_input = inject_noise(input_tensor, noise_level)
        noisy_image_path = os.path.join(output_dir, f"noisy_image_{noise_level}.jpeg")
        save_image_from_tensor(noisy_input, noisy_image_path)

        features.clear()
        with torch.no_grad():
            output = model(noisy_input)

        noisy_features = {name: features[layer].view(-1).numpy() for name, layer in model.named_modules() if name in layers_to_hook}

        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        top10_prob, top10_catid = torch.topk(probabilities, 10)

        classifications = [imagenet_classes[catid] for catid in top10_catid]

        noisy_features[f"classification_{noise_level}"] = classifications
        noisy_features[f"probability_{noise_level}"] = top10_prob.numpy()

        features_dict[f"noised_{noise_level}"] = noisy_features

    csv_path = os.path.join(output_dir, "features_with_gaussian_noise.csv")
    save_features_to_csv(features_dict, csv_path)
    print(f"Features and noisy images saved in {output_dir}")

if __name__ == "__main__":
    image_name = "n01753488_177"
    image_path = f"./Training/{image_name}.JPEG"
    output_dir = f"./output-{image_name}"
    imagenet_classes_path = "imagenet_classes.txt"
    os.makedirs(output_dir, exist_ok=True)
    process_gaussian_noise(image_path, output_dir, imagenet_classes_path)

