import os
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

class CustomImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, samples_per_class=10):
        super().__init__(root, transform)
        self.samples_per_class = samples_per_class
        self.filtered_samples = self._filter_samples()

    def _filter_samples(self):
        """Randomly select the specified number of samples."""
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
    """Extract features from ResNet18's layer1 and layer4."""
    def __init__(self, original_model):
        super(FeatureExtractor, self).__init__()
        self.original_model = original_model

    def forward(self, x):
        x = self.original_model.conv1(x)
        x = self.original_model.bn1(x)
        x = self.original_model.relu(x)
        x = self.original_model.maxpool(x)

        # layer1
        x1 = self.original_model.layer1(x)
        # layer2
        x2 = self.original_model.layer2(x1)
        # layer3
        x3 = self.original_model.layer3(x2)
        # layer4
        x4 = self.original_model.layer4(x3)
        return x1, x4

def gather_and_random_downsample(list_f1, list_f4):
    """
    Flatten and merge multiple batches of (layer1, layer4),
    then randomly downsample to equal length.
    """
    f1_array = np.concatenate(list_f1, axis=0)
    f4_array = np.concatenate(list_f4, axis=0)

    # Filter out zeros to avoid interference in visualization
    mask_f1 = (f1_array != 0)
    mask_f4 = (f4_array != 0)
    f1_nonzero = f1_array[mask_f1]
    f4_nonzero = f4_array[mask_f4]

    # Randomly downsample to the same length
    min_len = min(len(f1_nonzero), len(f4_nonzero))
    if min_len == 0:
        return np.array([]), np.array([])
    idx1 = np.random.choice(len(f1_nonzero), size=min_len, replace=False)
    idx4 = np.random.choice(len(f4_nonzero), size=min_len, replace=False)
    f1_sample = f1_nonzero[idx1]
    f4_sample = f4_nonzero[idx4]

    return f1_sample, f4_sample

def main():
    SAMPLES = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=True).to(device)
    model.eval()

    dataset_dir = "./Domain-Specific-Dataset/val"
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    
    dataset = CustomImageFolder(root=dataset_dir, transform=transform, samples_per_class=SAMPLES)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Prepare FGSM attacks (eps = 0.5 * e / 255, e in [0..9])
    fgsm_attacks = []
    for e in range(10):
        attacker = torchattacks.FGSM(model, eps=0.5*e/255.0)
        attacker.normalization_used = True
        attacker.set_normalization_used(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        fgsm_attacks.append(attacker)

    # Gaussian noise with sigma = 0.2*j, j in [0..9]
    gaussian_sigmas = [0.2 * i for i in range(10)]

    feature_extractor = FeatureExtractor(model.to(device))
    feature_extractor.eval()

    # Loop over 10 FGSM eps values and 10 Gaussian sigma values (100 combinations)
    for i in range(0,10):
        for j in range(0,10):
            eps_val = 0.5 * i / 255.0
            sigma_val = 0.2 * j

            fgsm_attacker = fgsm_attacks[i]
            gauss_sigma   = gaussian_sigmas[j]

            # Collect features from (Original, FGSM, Gaussian) for layer1 and layer4
            orig_layer1_list, orig_layer4_list = [], []
            fgsm_layer1_list, fgsm_layer4_list = [], []
            gauss_layer1_list, gauss_layer4_list = [], []

            # Process dataset, get features for original, FGSM, and Gaussian images
            for batch_images, batch_labels in dataloader:
                batch_images = batch_images.to(device)
                batch_labels = torch.tensor([int(lbl) for lbl in batch_labels], dtype=torch.long).to(device)

                # Original features
                with torch.no_grad():
                    f1_orig, f4_orig = feature_extractor(batch_images)
                orig_layer1_list.append(f1_orig.view(-1).cpu().numpy())
                orig_layer4_list.append(f4_orig.view(-1).cpu().numpy())

                # FGSM adversarial images (current eps)
                adv_images = fgsm_attacker(batch_images, batch_labels)
                with torch.no_grad():
                    f1_fgsm, f4_fgsm = feature_extractor(adv_images)
                fgsm_layer1_list.append(f1_fgsm.view(-1).cpu().numpy())
                fgsm_layer4_list.append(f4_fgsm.view(-1).cpu().numpy())

                # Gaussian noise images (current sigma)
                noise = torch.randn_like(batch_images) * gauss_sigma
                gauss_images = batch_images + noise
                with torch.no_grad():
                    f1_gauss, f4_gauss = feature_extractor(gauss_images)
                gauss_layer1_list.append(f1_gauss.view(-1).cpu().numpy())
                gauss_layer4_list.append(f4_gauss.view(-1).cpu().numpy())

            # Randomly downsample to same length for plotting
            orig_l1, orig_l4 = gather_and_random_downsample(orig_layer1_list, orig_layer4_list)
            fgsm_l1, fgsm_l4 = gather_and_random_downsample(fgsm_layer1_list, fgsm_layer4_list)
            gauss_l1, gauss_l4 = gather_and_random_downsample(gauss_layer1_list, gauss_layer4_list)

            # Plot 1x3 2D histograms
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(f"FGSM eps={i/2:.1f}/255, Gaussian sigma={0.2*j:.1f}", fontsize=14)

            # Original
            h1 = axs[0].hist2d(orig_l1, orig_l4, bins=50, range=[[0,3],[0,3]], cmap='viridis')
            axs[0].set_title("Original (layer1 vs. layer4)")
            axs[0].set_xlabel("Layer1 feature")
            axs[0].set_ylabel("Layer4 feature")
            fig.colorbar(h1[3], ax=axs[0])

            # FGSM
            h2 = axs[1].hist2d(fgsm_l1, fgsm_l4, bins=50, range=[[0,3],[0,3]], cmap='viridis')
            axs[1].set_title("FGSM (layer1 vs. layer4)")
            axs[1].set_xlabel("Layer1 feature")
            axs[1].set_ylabel("Layer4 feature")
            fig.colorbar(h2[3], ax=axs[1])

            # Gaussian
            h3 = axs[2].hist2d(gauss_l1, gauss_l4, bins=50, range=[[0,3],[0,3]], cmap='viridis')
            axs[2].set_title("Gaussian (layer1 vs. layer4)")
            axs[2].set_xlabel("Layer1 feature")
            axs[2].set_ylabel("Layer4 feature")
            fig.colorbar(h3[3], ax=axs[2])

            plt.tight_layout()
            plt.subplots_adjust(bottom=0.2)

            if len(orig_l1) > 0 and len(orig_l4) > 0:
                o1m, o1s, o1med = np.mean(orig_l1), np.std(orig_l1), np.median(orig_l1)
                o4m, o4s, o4med = np.mean(orig_l4), np.std(orig_l4), np.median(orig_l4)
            else:
                o1m = o1s = o1med = o4m = o4s = o4med = 0

            if len(fgsm_l1) > 0 and len(fgsm_l4) > 0:
                f1m, f1s, f1med = np.mean(fgsm_l1), np.std(fgsm_l1), np.median(fgsm_l1)
                f4m, f4s, f4med = np.mean(fgsm_l4), np.std(fgsm_l4), np.median(fgsm_l4)
            else:
                f1m = f1s = f1med = f4m = f4s = f4med = 0

            if len(gauss_l1) > 0 and len(gauss_l4) > 0:
                g1m, g1s, g1med = np.mean(gauss_l1), np.std(gauss_l1), np.median(gauss_l1)
                g4m, g4s, g4med = np.mean(gauss_l4), np.std(gauss_l4), np.median(gauss_l4)
            else:
                g1m = g1s = g1med = g4m = g4s = g4med = 0

            summary_text = (
                f"Original Layer1 mean={o1m:.4f}, std={o1s:.4f}, median={o1med:.4f}; "
                f"Layer4 mean={o4m:.4f}, std={o4s:.4f}, median={o4med:.4f}\n"
                f"FGSM     Layer1 mean={f1m:.4f}, std={f1s:.4f}, median={f1med:.4f}; "
                f"Layer4 mean={f4m:.4f}, std={f4s:.4f}, median={f4med:.4f}\n"
                f"Gaussian Layer1 mean={g1m:.4f}, std={g1s:.4f}, median={g1med:.4f}; "
                f"Layer4 mean={g4m:.4f}, std={g4s:.4f}, median={g4med:.4f}"
            )
            plt.figtext(0.5, 0, summary_text, ha='center', va='bottom', fontsize=11)

            out_filename = f"./2D-hist/2Dhist_eps_{i/2:.1f}_sigma_{0.2*j:.1f}.png"
            plt.savefig(out_filename, dpi=200)
            plt.close(fig)

if __name__ == "__main__":
    main()
