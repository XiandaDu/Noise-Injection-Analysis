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
        """Select random images from each class."""
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
    """Extract features from ResNet18 layers."""
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
        return x1, x2, x3, x4

def main():
    SAMPLES = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=True).to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    dataset_dir = "./Domain-Specific-Dataset/val"
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    
    dataset = CustomImageFolder(root=dataset_dir, transform=transform, samples_per_class=SAMPLES)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    fgsm_attacks = []
    for e in range(0, 10):
        attacker = torchattacks.FGSM(model, eps=0.5*e/255.0)
        attacker.normalization_used = True
        attacker.set_normalization_used(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        fgsm_attacks.append(attacker)

    gaussian_sigmas = [0.2 * i for i in range(0,10)]  

    all_fgsm_losses = [[] for _ in range(10)]  
    all_gaussian_losses = [[] for _ in range(10)]  

    all_fgsm_images = [[] for _ in range(10)]     
    all_gaussian_images = [[] for _ in range(10)] 

    sample_index = 0
    for batch_images, batch_labels in dataloader:
        batch_images = batch_images.to(device)
        batch_labels = torch.tensor([int(label) for label in batch_labels], dtype=torch.long).to(device)

        for i, attacker in enumerate(fgsm_attacks):
            if i == 0:
                adv_images = batch_images  
            else:
                adv_images = attacker(batch_images, batch_labels)
            adv_output = model(adv_images)
            adv_loss = criterion(adv_output, batch_labels)
            all_fgsm_losses[i].append(adv_loss.item())
            all_fgsm_images[i].append(adv_images.detach().cpu())

        for j, sigma in enumerate(gaussian_sigmas):
            noise = torch.randn_like(batch_images) * sigma
            noisy_images = torch.clamp(batch_images + noise, -1000, 1000)
            noisy_output = model(noisy_images)
            noisy_loss = criterion(noisy_output, batch_labels)
            all_gaussian_losses[j].append(noisy_loss.item())
            all_gaussian_images[j].append(noisy_images.detach().cpu())

        sample_index += 1

    fgsm_avg = [np.mean(loss_list) for loss_list in all_fgsm_losses]
    gauss_avg = [np.mean(loss_list) for loss_list in all_gaussian_losses]

    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.plot(range(0,10), fgsm_avg, marker='o')
    plt.title("FGSM Loss vs. eps=0.5...5/255")
    plt.xlabel("eps")
    plt.ylabel("CE Loss")

    plt.subplot(1,2,2)
    plt.plot([0.1*i for i in range(0,10)], gauss_avg, marker='o')
    plt.title("Gaussian Loss vs. sigma=0.2...2")
    plt.xlabel("sigma")
    plt.ylabel("CE Loss")
    plt.tight_layout()
    plt.show()

    feature_extractor = FeatureExtractor(model.to(device))
    feature_extractor.eval()

    layer1_dict = {(i,j): [] for i in range(10) for j in range(10)}
    layer2_dict = {(i,j): [] for i in range(10) for j in range(10)}
    layer3_dict = {(i,j): [] for i in range(10) for j in range(10)}
    layer4_dict = {(i,j): [] for i in range(10) for j in range(10)}

    num_samples = len(dataset)

    for k in range(num_samples):
        for i in range(10):
            loss_fgsm = all_fgsm_losses[i][k]
            fgsm_img  = all_fgsm_images[i][k].to(device)  
            for j in range(10):
                loss_gauss = all_gaussian_losses[j][k]
                if abs(loss_fgsm - loss_gauss) < 0.3:
                    gauss_img = all_gaussian_images[j][k].to(device)
                    with torch.no_grad():
                        f1_fgsm, f2_fgsm, f3_fgsm, f4_fgsm = feature_extractor(fgsm_img)
                        f1_gauss, f2_gauss, f3_gauss, f4_gauss = feature_extractor(gauss_img)

                    f1_fgsm = f1_fgsm.view(-1).cpu().numpy()
                    f1_gauss= f1_gauss.view(-1).cpu().numpy()
                    f2_fgsm = f2_fgsm.view(-1).cpu().numpy()
                    f2_gauss= f2_gauss.view(-1).cpu().numpy()
                    f3_fgsm = f3_fgsm.view(-1).cpu().numpy()
                    f3_gauss= f3_gauss.view(-1).cpu().numpy()
                    f4_fgsm = f4_fgsm.view(-1).cpu().numpy()
                    f4_gauss= f4_gauss.view(-1).cpu().numpy()

                    layer1_dict[(i,j)].append((f1_fgsm, f1_gauss))
                    layer2_dict[(i,j)].append((f2_fgsm, f2_gauss))
                    layer3_dict[(i,j)].append((f3_fgsm, f3_gauss))
                    layer4_dict[(i,j)].append((f4_fgsm, f4_gauss))

    def gather_and_filter(data_pairs):
        """Merge and filter data, removing (0,0) points."""
        x_vals, y_vals = [], []
        for (f_arr, g_arr) in data_pairs:
            x_vals.extend(f_arr)
            y_vals.extend(g_arr)
        x_vals = np.array(x_vals)
        y_vals = np.array(y_vals)

        mask = ~((x_vals == 0) | (y_vals == 0))
        return x_vals[mask], y_vals[mask]

    for i in range(10):
        eps_val = f"0.5*{i}/255"
        for j in range(10):
            sigma_val = f"0.2*{j}"
            l1_pairs = layer1_dict[(i,j)]
            l2_pairs = layer2_dict[(i,j)]
            l3_pairs = layer3_dict[(i,j)]
            l4_pairs = layer4_dict[(i,j)]

            if len(l1_pairs) == 0:
                continue

            l1_x, l1_y = gather_and_filter(l1_pairs)
            l2_x, l2_y = gather_and_filter(l2_pairs)
            l3_x, l3_y = gather_and_filter(l3_pairs)
            l4_x, l4_y = gather_and_filter(l4_pairs)

            if len(l1_x) == 0:
                continue

            fig, axs = plt.subplots(2, 2, figsize=(10, 8))
            fig.suptitle(f"FGSM eps={eps_val}  vs  Gaussian sigma={sigma_val}", fontsize=14)

            h1 = axs[0,0].hist2d(l1_x, l1_y, bins=50, range=[[0,1],[0,1]], cmap='viridis')
            axs[0,0].set_title("Layer1 Features")
            fig.colorbar(h1[3], ax=axs[0,0])

            h2 = axs[0,1].hist2d(l2_x, l2_y, bins=50, range=[[0,1],[0,1]], cmap='viridis')
            axs[0,1].set_title("Layer2 Features")
            fig.colorbar(h2[3], ax=axs[0,1])

            h3 = axs[1,0].hist2d(l3_x, l3_y, bins=50, range=[[0,1],[0,1]], cmap='viridis')
            axs[1,0].set_title("Layer3 Features")
            fig.colorbar(h3[3], ax=axs[1,0])

            h4 = axs[1,1].hist2d(l4_x, l4_y, bins=50, range=[[0,1],[0,1]], cmap='viridis')
            axs[1,1].set_title("Layer4 Features")
            fig.colorbar(h4[3], ax=axs[1,1])

            for ax in axs.flat:
                ax.set_xlabel("FGSM feature value")
                ax.set_ylabel("Gaussian feature value")

            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    main()
