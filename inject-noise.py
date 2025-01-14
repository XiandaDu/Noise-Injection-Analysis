import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import random

model = models.resnet18(weights=True)
model.eval()

features = {}

def hook(module, input, output):
    features[module] = output

layers_to_hook = ["layer1", "layer2", "layer3", "layer4"]
for name, layer in model.named_modules():
    if name in layers_to_hook:
        layer.register_forward_hook(hook)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image_path = "face-1"
image = Image.open(f"{image_path}.jpg").convert("RGB")

input_tensor = transform(image).unsqueeze(0)

def inject_noise(image_tensor, noise_level=0.2):
    noise = torch.randn_like(image_tensor) * noise_level
    return image_tensor + noise

def inject_dot_noise(image_tensor, noise_percentage=0.05):
    image_np = image_tensor.squeeze(0).permute(1, 2, 0).detach().numpy()
    h, w, c = image_np.shape
    total_pixels = h * w
    num_noisy_pixels = int(total_pixels * noise_percentage)

    for _ in range(num_noisy_pixels):
        x = random.randint(0, h - 1)
        y = random.randint(0, w - 1)
        image_np[x, y, :] = 1

    noisy_tensor = torch.tensor(image_np).permute(2, 0, 1).unsqueeze(0)
    return noisy_tensor

def save_features_to_csv(original, gaussian, dot, file_name):
    layer_columns = []
    for i, layer_name in enumerate(layers_to_hook):
        original_layer = original[layer_name].view(-1).numpy()
        gaussian_layer = gaussian[layer_name].view(-1).numpy()
        dot_layer = dot[layer_name].view(-1).numpy()

        layer_columns.extend([
            pd.Series(original_layer, name=f"{layer_name}_original"),
            pd.Series(gaussian_layer, name=f"{layer_name}_gaussian"),
            pd.Series(dot_layer, name=f"{layer_name}_dot"),
        ])

    df = pd.concat(layer_columns, axis=1)
    df.to_csv(file_name, index=False)
    print(f"Saved features to {file_name}")

def save_image_from_tensor(image_tensor, file_name):
    unnormalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )
    unnormalized_tensor = unnormalize(image_tensor.squeeze(0))
    clipped_tensor = torch.clamp(unnormalized_tensor, 0, 1)
    noisy_image = transforms.ToPILImage()(clipped_tensor)
    noisy_image.save(file_name)
    print(f"Noisy image saved to {file_name}")

with torch.no_grad():
    model(input_tensor)

original_features = {name: features[layer] for name, layer in model.named_modules() if name in layers_to_hook}

noisy_input_gaussian = inject_noise(input_tensor, noise_level=0.2)
save_image_from_tensor(noisy_input_gaussian, f"{image_path}-gaussian-noised.jpg")
features.clear()
with torch.no_grad():
    model(noisy_input_gaussian)

noisy_features_gaussian = {name: features[layer] for name, layer in model.named_modules() if name in layers_to_hook}

noisy_input_dot = inject_dot_noise(input_tensor, noise_percentage=0.05)
save_image_from_tensor(noisy_input_dot, f"{image_path}-dot-noised.jpg")
features.clear()
with torch.no_grad():
    model(noisy_input_dot)

noisy_features_dot = {name: features[layer] for name, layer in model.named_modules() if name in layers_to_hook}

save_features_to_csv(original_features, noisy_features_gaussian, noisy_features_dot, f"{image_path}_features.csv")
