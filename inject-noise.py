import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd

model = models.resnet18(weights=True)
model.eval()
features = {}

def hook(module, input, output):
    features[module] = output

for name, layer in model.named_modules():
    if name == "layer1":
        layer.register_forward_hook(hook)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image_path = "face-1.jpg"
image = Image.open(image_path).convert("RGB")

input_tensor = transform(image).unsqueeze(0)

def inject_noise(image_tensor, noise_level=0.2):
    noise = torch.randn_like(image_tensor) * noise_level
    return image_tensor + noise

def save_features_to_csv(features, file_name_prefix):
    for layer, feature in features.items():
        feature_np = feature.squeeze().detach().numpy()
        feature_flat = feature_np.reshape(-1)
        column_name = f"{file_name_prefix}_features"
        df = pd.DataFrame(feature_flat, columns=[column_name])
        csv_file = f"{file_name_prefix}.csv"
        df.to_csv(csv_file, index=False)
        print(f"Saved features for {layer} to {csv_file}")

with torch.no_grad():
    model(input_tensor)

original_features = {k: v.clone() for k, v in features.items()}

noisy_input = inject_noise(input_tensor, noise_level=0.2)
features.clear()
with torch.no_grad():
    model(noisy_input)

noisy_features = {k: v.clone() for k, v in features.items()}

save_features_to_csv(original_features, "original_features")
save_features_to_csv(noisy_features, "noisy_features")

for layer, original_feat in original_features.items():
    noisy_feat = noisy_features[layer]
    diff = torch.mean((original_feat - noisy_feat) ** 2).item()
    print(f"Layer: {layer}, Feature Change (MSE): {diff}")
