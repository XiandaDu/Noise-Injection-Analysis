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


def save_predictions_comparison_to_csv(original_preds, perturbed_preds_2, perturbed_preds_4, perturbed_preds_6, perturbed_preds_8, filename, imagenet_classes):
    data = {
        "Original Class": [imagenet_classes[idx] for idx in original_preds["indices"]],
        "Original Probability": [f"{prob:.4f}" for prob in original_preds["probabilities"]],
        "Adversarial Class ESP@2": [imagenet_classes[idx] for idx in perturbed_preds_2["indices"]],
        "Adversarial Probability ESP@2": [f"{prob:.4f}" for prob in perturbed_preds_2["probabilities"]],
        "Adversarial Class ESP@4": [imagenet_classes[idx] for idx in perturbed_preds_4["indices"]],
        "Adversarial Probability ESP@4": [f"{prob:.4f}" for prob in perturbed_preds_4["probabilities"]],
        "Adversarial Class ESP@6": [imagenet_classes[idx] for idx in perturbed_preds_6["indices"]],
        "Adversarial Probability ESP@6": [f"{prob:.4f}" for prob in perturbed_preds_6["probabilities"]],
        "Adversarial Class ESP@8": [imagenet_classes[idx] for idx in perturbed_preds_8["indices"]],
        "Adversarial Probability ESP@8": [f"{prob:.4f}" for prob in perturbed_preds_8["probabilities"]],
    }
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

EPS_VAL = 2
imagenet_path = "./Domain-Specific-Dataset/val"
output_dir = "./fgsm_result"
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

attack_2 = torchattacks.FGSM(model, eps=EPS_VAL/255)
attack_2.normalization_used = True
attack_2.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
EPS_VAL += 2

attack_4 = torchattacks.FGSM(model, eps=EPS_VAL/255)
attack_4.normalization_used = True
attack_4.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
EPS_VAL += 2

attack_6 = torchattacks.FGSM(model, eps=EPS_VAL/255)
attack_6.normalization_used = True
attack_6.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
EPS_VAL += 2

attack_8 = torchattacks.FGSM(model, eps=EPS_VAL/255)
attack_8.normalization_used = True
attack_8.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

extractor = FeatureExtractor(model)
extractor.register_hooks()

with open("imagenet_classes.txt", "r") as f:
    imagenet_classes = [line.strip() for line in f]


repeat = 5
i = 0
for i, (images, labels) in enumerate(tqdm(dataloader)):
    adv_images_2 = attack_2(images, labels)
    adv_images_4 = attack_4(images, labels)
    adv_images_6 = attack_6(images, labels)
    adv_images_8 = attack_8(images, labels)

    logits_original = model(images)
    original_predictions = get_top_predictions_with_labels(logits_original)

    logits_adversarial_2 = model(adv_images_2)
    perturbed_predictions_2 = get_top_predictions_with_labels(logits_adversarial_2)

    logits_adversarial_4 = model(adv_images_4)
    perturbed_predictions_4 = get_top_predictions_with_labels(logits_adversarial_4)

    logits_adversarial_6 = model(adv_images_6)
    perturbed_predictions_6 = get_top_predictions_with_labels(logits_adversarial_6)

    logits_adversarial_8 = model(adv_images_8)
    perturbed_predictions_8 = get_top_predictions_with_labels(logits_adversarial_8)

    save_image_from_tensor(images, os.path.join(output_dir, f"original_{i + 1}.png"))
    save_image_from_tensor(adv_images_2, os.path.join(output_dir, f"fgsm_{i + 1}_e2.png"))
    save_image_from_tensor(adv_images_4, os.path.join(output_dir, f"fgsm_{i + 1}_e4.png"))
    save_image_from_tensor(adv_images_6, os.path.join(output_dir, f"fgsm_{i + 1}_e6.png"))
    save_image_from_tensor(adv_images_8, os.path.join(output_dir, f"fgsm_{i + 1}_e8.png"))

    save_predictions_comparison_to_csv(
        original_predictions,
        perturbed_predictions_2,
        perturbed_predictions_4,
        perturbed_predictions_6,
        perturbed_predictions_8,
        os.path.join(output_dir, f"predictions_{i + 1}.csv"),
        imagenet_classes
    )

    extractor.clear_features()
    model(images)
    save_feature_maps_to_csv(extractor.features, os.path.join(output_dir, f"original_features_{i + 1}.csv"))

    extractor.clear_features()
    model(adv_images_2)
    save_feature_maps_to_csv(extractor.features, os.path.join(output_dir, f"fgsm_features_{i + 1}_e2.csv"))

    extractor.clear_features()
    model(adv_images_4)
    save_feature_maps_to_csv(extractor.features, os.path.join(output_dir, f"fgsm_features_{i + 1}_e4.csv"))

    extractor.clear_features()
    model(adv_images_6)
    save_feature_maps_to_csv(extractor.features, os.path.join(output_dir, f"fgsm_features_{i + 1}_e6.csv"))

    extractor.clear_features()
    model(adv_images_8)
    save_feature_maps_to_csv(extractor.features, os.path.join(output_dir, f"fgsm_features_{i + 1}_e8.csv"))

    i = i+1
    if(i>repeat):
        break
