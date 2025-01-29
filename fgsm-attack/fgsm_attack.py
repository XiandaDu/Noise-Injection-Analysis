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
    def __init__(self, model, layers_to_hook=None):
        self.model = model
        self.features = {}
        self.hooks = []
        self.layers_to_hook = layers_to_hook or ["layer1", "layer2", "layer3", "layer4"]

    def _hook_fn(self, module, input, output):
        self.features[module] = output

    def register_hooks(self):
        for name, layer in self.model.named_modules():
            if name in self.layers_to_hook:
                self.hooks.append(layer.register_forward_hook(self._hook_fn))

    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def clear_features(self):
        self.features.clear()


def save_feature_maps_to_csv(features, filename):
    data = {f"Layer_{i + 1}": feature_map.detach().cpu().numpy().flatten()
            for i, feature_map in enumerate(features.values())}

    max_length = max(len(v) for v in data.values())
    for key, value in data.items():
        if len(value) < max_length:
            data[key] = list(value) + [float('nan')] * (max_length - len(value))

    pd.DataFrame(data).to_csv(filename, index=False)
    print(f"Saved: {filename}")

def save_feature_maps_to_csv_by_depth(features, filename):
    data = {}
    max_length = 0

    i = 1
    for layer_name, layer_data in features.items():
        for depth in range(layer_data.shape[1]):
            feature_values = layer_data[0, depth].view(-1).detach().numpy()
            max_length = max(max_length, len(feature_values))
            data[f"Layer_{i}_depth{depth}"] = feature_values
        i += 1

    for key, value in data.items():
        if len(value) < max_length:
            data[key] = list(value) + [float('nan')] * (max_length - len(value))

    pd.DataFrame(data).to_csv(filename, index=False)
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

def save_predictions_comparison_to_csv(original_preds, perturbed_preds_list, filename, imagenet_classes):
    data = {
        "Original Class": [imagenet_classes[idx] for idx in original_preds["indices"]],
        "Original Probability": [f"{prob:.4f}" for prob in original_preds["probabilities"]]
    }

    for esp, perturbed_preds in enumerate(perturbed_preds_list):
        data[f"Adversarial Class ESP@{esp+1}"] = [imagenet_classes[idx] for idx in perturbed_preds["indices"]]
        data[f"Adversarial Probability ESP@{esp+1}"] = [f"{prob:.4f}" for prob in perturbed_preds["probabilities"]]

    pd.DataFrame(data).to_csv(filename, index=False)
    print(f"Prediction comparison saved to {filename}")


def get_top_predictions_with_labels(logits, top_k=10):
    probabilities = torch.softmax(logits, dim=1)
    top_probs, top_indices = torch.topk(probabilities, top_k)
    return {
        "indices": top_indices.squeeze().tolist(),
        "probabilities": top_probs.squeeze().tolist(),
    }


model = models.resnet18(pretrained=True)
model.eval()

EPS_VAL = 1
ITERATION = 9
imagenet_path = "../Domain-Specific-Dataset/val"
output_dir = "./fgsm_results"
os.makedirs(output_dir, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
imagenet_val_dataset = ImageFolder(root=imagenet_path, transform=transform)
dataloader = DataLoader(imagenet_val_dataset, batch_size=1, shuffle=True)

attack = []
for i in range(0, ITERATION):
    attack.append(torchattacks.FGSM(model, eps=EPS_VAL/255))
    attack[i].normalization_used = True
    attack[i].set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    EPS_VAL += 1

extractor = FeatureExtractor(model)
extractor.register_hooks()

with open("imagenet_classes.txt", "r") as f:
    imagenet_classes = [line.strip() for line in f]


repeat = 5
i = 0
for i, (images, labels) in enumerate(tqdm(dataloader)):
    os.makedirs(f"./fgsm_results/image{i+1}", exist_ok=True)

    adv_images = []
    for idx in range(0, ITERATION):
        adv_images.append(attack[idx](images, labels))

    logits_original = model(images)
    original_predictions = get_top_predictions_with_labels(logits_original)

    logits_adversarial = []
    perturbed_predictions = []
    for idx in range(0, ITERATION):
        logits_adversarial.append(model(adv_images[idx]))
        perturbed_predictions.append(get_top_predictions_with_labels(logits_adversarial[idx]))

    save_image_from_tensor(images, os.path.join(output_dir+f"/image{i+1}", f"orriginal_{i + 1}_e0.png"))
    for idx in range(0, ITERATION):
        save_image_from_tensor(adv_images[idx], os.path.join(output_dir+f"/image{i+1}", f"fgsm_{i + 1}_e{idx + 1}.png"))

    save_predictions_comparison_to_csv(
        original_predictions,
        perturbed_predictions,
        os.path.join(output_dir+f"/image{i+1}", f"predictions_{i + 1}.csv"),
        imagenet_classes
    )

    extractor.clear_features()
    model(images)
    save_feature_maps_to_csv(extractor.features, os.path.join(output_dir+f"/image{i+1}", f"original_features_{i + 1}_e0.csv"))

    for idx in range(0, ITERATION):
        extractor.clear_features()
        model(adv_images[idx])
        save_feature_maps_to_csv(extractor.features, os.path.join(output_dir+f"/image{i+1}", f"fgsm_features_{i + 1}_e{idx + 1}.csv"))

    os.makedirs(f"./fgsm_results/image{i+1}-by-depth", exist_ok=True)
    for idx in range(0, ITERATION):
        extractor.clear_features()
        model(adv_images[idx])
        save_feature_maps_to_csv_by_depth(extractor.features, os.path.join(output_dir+f"/image{i+1}-by-depth", f"fgsm_features_{i + 1}_e{idx + 1}.csv"))

    i = i+1
    if(i>repeat):
        break
