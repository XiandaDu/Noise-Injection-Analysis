import random 
import torch
import torch.nn as nn
import torchattacks
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np

class ResNet18_Modified(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.layer2 = original_model.layer2
        self.layer3 = original_model.layer3
        self.layer4 = original_model.layer4
        self.avgpool = original_model.avgpool
        self.fc = original_model.fc

    def forward(self, x):
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
class FeatureExtractorModified(nn.Module):
    def __init__(self, original_model):
        super(FeatureExtractorModified, self).__init__()
        self.original_model = original_model

    def forward(self, x):
        x2 = self.original_model.layer2(x)
        x3 = self.original_model.layer3(x2)
        x4 = self.original_model.layer4(x3)
        return x4

class CustomImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, samples_per_class=10):
        super().__init__(root, transform)
        self.samples_per_class = samples_per_class
        self.filtered_samples = self._filter_samples()

    def _filter_samples(self):
        class_to_samples = {}
        for path, label in self.samples:
            class_name = self.classes[label]
            if class_name not in class_to_samples:
                class_to_samples[class_name] = []
            class_to_samples[class_name].append((path, class_name))

        selected_samples = []
        for samples in class_to_samples.values():
            selected_samples.extend(random.sample(samples, min(len(samples), self.samples_per_class)))

        return selected_samples

    def __len__(self):
        return len(self.filtered_samples)

    def __getitem__(self, idx):
        path, label_name = self.filtered_samples[idx]
        sample = self.loader(path)
        if self.transform:
            sample = self.transform(sample)
        return sample, label_name

class FeatureExtractor(nn.Module):
    def __init__(self, original_model):
        super(FeatureExtractor, self).__init__()
        self.original_model = original_model

    def forward(self, x):
        x = self.original_model.conv1(x)
        x = self.original_model.bn1(x)
        x = self.original_model.relu(x)
        x = self.original_model.maxpool(x)

        x1 = self.original_model.layer1(x)
        x2 = self.original_model.layer2(x1)
        x3 = self.original_model.layer3(x2)
        x4 = self.original_model.layer4(x3)
        return x1, x4

def gather_and_random_downsample(list_f1, list_f4):
    f1_array = np.concatenate(list_f1, axis=0)
    f4_array = np.concatenate(list_f4, axis=0)

    f1_nonzero = f1_array[f1_array != 0]
    f4_nonzero = f4_array[f4_array != 0]

    percentile_f1 = np.percentile(f1_nonzero, 99.5)
    percentile_f4 = np.percentile(f4_nonzero, 99.5)

    mask_f1 = (f1_array != 0) & (f1_array <= percentile_f1)
    mask_f4 = (f4_array != 0) & (f4_array <= percentile_f4)

    f1_filtered = f1_array[mask_f1]
    f4_filtered = f4_array[mask_f4]

    min_len = min(len(f1_filtered), len(f4_filtered))
    if min_len == 0:
        return np.array([]), np.array([])

    idx1 = np.random.choice(len(f1_filtered), size=min_len, replace=False)
    idx4 = np.random.choice(len(f4_filtered), size=min_len, replace=False)

    f1_sample = f1_filtered[idx1]
    f4_sample = f4_filtered[idx4]

    return f1_sample, f4_sample

from torchattacks.attack import Attack

class FGSM_MODIFIED(Attack):
    def __init__(self, model, eps=8 / 255):
        super().__init__("FGSM", model)
        self.eps = eps
        self.supported_mode = ["default", "targeted"]

    def forward(self, images, labels, max_mag):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()
        images.requires_grad = True
        outputs = self.get_logits(images)

        if self.targeted:
            cost = -loss(outputs, target_labels)
        else:
            cost = loss(outputs, labels)

        grad = torch.autograd.grad(cost, images, retain_graph=False, create_graph=False)[0]
        adv_images = images + self.eps * grad.sign()
        adv_images = torch.clamp(adv_images, min=0, max=max_mag).detach()

        return adv_images

def main():
    SAMPLES = 40
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=True).to(device)
    model.eval()

    model_modified = ResNet18_Modified(models.resnet18(pretrained=True)).to(device)
    model_modified.eval()

    dataset_dir = "./Domain-Specific-Dataset/val"
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    dataset = CustomImageFolder(root=dataset_dir, transform=transform, samples_per_class=SAMPLES)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    fgsm_attacks = []
    for e in range(10):
        attacker = torchattacks.FGSM(model, eps=0.5*e/255.0)
        attacker.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        fgsm_attacks.append(attacker)

    feature_extractor = FeatureExtractor(model.to(device))
    feature_extractor.eval()
    feature_extractor_modified = FeatureExtractorModified(model_modified.to(device))
    feature_extractor_modified.eval()

    for i in range(0,10):
        fgsm_attacker = fgsm_attacks[i]

        fgsm_loss_list = []
        eps_grid = np.arange(0, 15, 0.5)
        layer4_loss_sum = {e: 0.0 for e in eps_grid}
        layer4_loss_count = {e: 0 for e in eps_grid}
        list_f1, list_f4, list_orig1, list_orig4 = [], [], [], []

        list_list_modified4 = {float(e): [] for e in eps_grid}

        for batch_images, batch_labels in dataloader:
            batch_images = batch_images.to(device)
            batch_images.requires_grad = True
            batch_labels = torch.tensor([int(lbl) for lbl in batch_labels], dtype=torch.long).to(device)

            with torch.no_grad():
                f1_orig, f4_orig = feature_extractor(batch_images)
            list_orig1.append(f1_orig.view(-1).cpu().numpy())
            list_orig4.append(f4_orig.view(-1).cpu().numpy())
            f1_orig.requires_grad = True

            adv_images = fgsm_attacker(batch_images, batch_labels)
            with torch.no_grad():
                f1_adv, f4_adv = feature_extractor(adv_images)
            list_f1.append(f1_adv.view(-1).cpu().numpy())
            list_f4.append(f4_adv.view(-1).cpu().numpy())

            fgsm_output = model(adv_images)
            fgsm_loss = criterion(fgsm_output, batch_labels)
            fgsm_loss_list.append(fgsm_loss.item())

            fixed_input = f1_orig.detach()

            for e4 in eps_grid:
                attacker_layer4 = FGSM_MODIFIED(model_modified, eps=e4/255.0)
                max_mag = fixed_input.max()
                modified_layer4 = attacker_layer4.forward(fixed_input, batch_labels, max_mag)
                output_layer4 = model_modified(modified_layer4)
                with torch.no_grad():
                    modified4 = feature_extractor_modified(modified_layer4)
                loss_layer4 = criterion(output_layer4, batch_labels)
                layer4_loss_sum[e4] += loss_layer4.item()
                layer4_loss_count[e4] += 1

                f4_np = modified4.view(-1).cpu().numpy()
                list_list_modified4[float(e4)].append(f4_np)

        eps_list = []
        loss_avg_list = []
        for e in eps_grid:
            if layer4_loss_count[e] > 0:
                eps_list.append(e)
                loss_avg_list.append(layer4_loss_sum[e] / layer4_loss_count[e])

        fgsm_loss_avg = sum(fgsm_loss_list) / len(fgsm_loss_list)

        plt.figure(figsize=(7,5))
        plt.plot(eps_list, loss_avg_list, color='blue', label='Layer4 epsilon sweep (avg)')
        plt.axhline(y=fgsm_loss_avg, color='red', linestyle='--', label=f'Layer1 CE loss={fgsm_loss_avg:.4f}')
        plt.xlabel("Layer4 epsilon")
        plt.ylabel("Cross Entropy Loss")
        plt.title(f"Avg Loss Curve | Layer1 eps={0.5*i}/255")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"./2D-hist/L4_loss_avg_curve_layer1eps_{0.5*i:.1f}.png", dpi=200)
        plt.close()

        fgsm_loss_array = np.array(fgsm_loss_list)
        target_loss = np.mean(fgsm_loss_array)

        closest_eps = min(eps_list, key=lambda e: abs((layer4_loss_sum[e]/layer4_loss_count[e]) - target_loss))
        print(f"[INFO] Layer1 eps={0.5*i}/255, matched Layer4 eps={closest_eps}/255")

        list_modified4 = list_list_modified4[closest_eps]
        flat_orig1, flat_f1 = gather_and_random_downsample(list_orig1, list_f1)
        flat_orig4, flat_f4 = gather_and_random_downsample(list_orig4, list_f4)
        list_modified4, _ = gather_and_random_downsample(list_modified4, list_modified4)

        if len(flat_orig1) > 0:
            plt.figure(figsize=(10, 6))
            plt.hist(flat_orig1, bins=100, alpha=0.5, label="Original Layer1", density=True)
            plt.hist(flat_f1, bins=100, alpha=0.5, label=f"FGSM Layer1 eps={0.5*i}/255", density=True)
            plt.xlabel("Feature Value")
            plt.ylabel("Density")
            plt.title(f"Feature Distribution | Layer1 Injection eps={0.5*i}/255")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"./2D-hist/L1_feature_hist_{i}.png", dpi=200)
            plt.close()

        if len(flat_orig4) > 0:
            plt.figure(figsize=(10, 6))
            plt.hist(flat_orig4, bins=100, alpha=0.5, label="Original Layer4", density=True)
            plt.hist(flat_f4, bins=100, alpha=0.5, label=f"Layer 1 FGSM @Layer4 eps={0.5*i}/255", density=True)
            plt.hist(list_modified4, bins=100, alpha=0.5, label=f"Layer4 FGSM @ Layer4 eps={closest_eps}/255", density=True)
            plt.xlabel("Feature Value")
            plt.ylabel("Density")
            plt.title(f"Feature Distribution | L1 esp={0.5*i}/255 matched L4 eps={closest_eps}/255")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"./2D-hist/L4_feature_hist_{i}.png", dpi=200)
            plt.close()

if __name__ == "__main__":
    main()
