import torch
import os
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
import torchattacks

class FeatureExtractor:
    def __init__(self, model):
        self.model = model
        self.features = {}
        self.hooks = []

    def _hook_fn(self, module, input, output):
        self.features[module] = output

    def register_hooks(self):
        layers_to_hook = ["layer1", "layer2", "layer3", "layer4"]
        for name, layer in self.model.named_modules():
            if name in layers_to_hook:
                self.hooks.append(layer.register_forward_hook(self._hook_fn))

    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def clear_features(self):
        self.features.clear()

model = models.resnet18(pretrained=True)
model.eval()

EPS_VAL = 8/255
imagenet_path = "./Domain-Specific-Dataset"
val_dir = f"{imagenet_path}/val"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

imagenet_val_dataset = ImageFolder(root=val_dir, transform=transform)
dataloader = DataLoader(imagenet_val_dataset, batch_size=1, shuffle=True)


attack = torchattacks.FGSM(model, eps=EPS_VAL)

extractor = FeatureExtractor(model)
extractor.register_hooks()

def save_feature_maps_to_csv(features, filename):
    data = {}
    for i, (layer, feature_map) in enumerate(features.items()):
        flattened = feature_map.detach().cpu().numpy().flatten()
        data[f"Layer_{i + 1}"] = flattened
    max_length = max(len(v) for v in data.values())

    for key, value in data.items():
        if len(value) < max_length:
            data[key] = list(value) + [float('nan')] * (max_length - len(value))

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Saved: {filename}")


def save_image_from_tensor(image_tensor, file_name):
    unnormalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )

    unnormalized_tensor = unnormalize(image_tensor.squeeze(0))
    clipped_tensor = torch.clamp(unnormalized_tensor, 0, 1)

    noisy_image = transforms.ToPILImage()(clipped_tensor)
    noisy_image.save(file_name)
    print(f"Image saved to {file_name}")


output_dir = "./fgsm_result"
os.makedirs(output_dir, exist_ok=True)

with open("imagenet_classes.txt", "r") as f:
    imagenet_classes = [line.strip() for line in f]


def save_predictions_comparison_to_csv(original_preds, perturbed_preds, filename):
    data = {
        "Original Class": [
            imagenet_classes[int(idx)] for idx in original_preds["indices"]
        ],
        "Original Probability": [
            f"{prob:.4f}" for prob in original_preds["probabilities"]
        ],
        "Adversarial Class": [
            imagenet_classes[int(idx)] for idx in perturbed_preds["indices"]
        ],
        "Adversarial Probability": [
            f"{prob:.4f}" for prob in perturbed_preds["probabilities"]
        ],
    }
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Prediction comparison saved to {filename}")


def get_top_predictions_with_labels(logits, top_k=10):
    probabilities = torch.softmax(logits, dim=1)
    top_probs, top_indices = torch.topk(probabilities, top_k)
    return {
        "indices": top_indices.squeeze().tolist(),
        "probabilities": top_probs.squeeze().tolist(),
    }


# Main Loop
for i, (images, labels) in enumerate(tqdm(dataloader)):
    # FGSM
    adv_images = attack(images, labels)

    original_image_path = os.path.join(output_dir, f"original_{i+1}.png")
    adversarial_image_path = os.path.join(output_dir, f"fgsm_{i+1}.png")
    save_image_from_tensor(images, original_image_path)
    save_image_from_tensor(adv_images, adversarial_image_path)

    # Prediction of the original pic
    logits_original = model(images)
    original_predictions = get_top_predictions_with_labels(logits_original)

    # Prediction of the adv pic
    logits_adversarial = model(adv_images)
    perturbed_predictions = get_top_predictions_with_labels(logits_adversarial)

    predictions_csv_path = os.path.join(output_dir, f"predictions_{i+1}.csv")
    save_predictions_comparison_to_csv(original_predictions, perturbed_predictions, filename=predictions_csv_path)

    # Original Feature Maps
    extractor.clear_features()
    model(images)
    original_features = extractor.features.copy()
    save_feature_maps_to_csv(original_features, filename=os.path.join(output_dir, f"original_features_{i+1}.csv"))

    # Adv Feature Maps
    extractor.clear_features()
    model(adv_images)
    perturbed_features = extractor.features.copy()
    save_feature_maps_to_csv(perturbed_features, filename=os.path.join(output_dir, f"fgsm_features_{i+1}.csv"))

    break