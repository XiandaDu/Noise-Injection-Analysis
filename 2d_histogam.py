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
        """Select only `samples_per_class` random images from each class (subfolder)"""
        class_to_samples = {}
        for path, label in self.samples:
            class_name = self.classes[label]  # Get the class name (i.e., folder name)
            if class_name not in class_to_samples:
                class_to_samples[class_name] = []
            class_to_samples[class_name].append((path, class_name))  # Store class name

        selected_samples = []
        for samples in class_to_samples.values():
            selected_samples.extend(random.sample(samples, min(len(samples), self.samples_per_class)))

        return selected_samples

    def __len__(self):
        return len(self.filtered_samples)

    def __getitem__(self, idx):
        path, label_name = self.filtered_samples[idx]  # Here, label_name is the class name
        sample = self.loader(path)
        if self.transform:
            sample = self.transform(sample)
        return sample, label_name  # Directly return the class name



def main():
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
    
    # Only iterate over 10 images per class
    dataset = CustomImageFolder(root=dataset_dir, transform=transform, samples_per_class=10)
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

    for batch_images, batch_labels in dataloader:
        batch_images = batch_images.to(device)
        batch_labels = torch.tensor([int(label) for label in batch_labels], dtype=torch.long).to(device)

        for i, attacker in enumerate(fgsm_attacks):
            if i == 0:
                adv_images = batch_images  # Use original image when eps=0
            else:
                adv_images = attacker(batch_images, batch_labels)
            adv_output = model(adv_images)
            adv_loss = criterion(adv_output, batch_labels)
            all_fgsm_losses[i].append(adv_loss.item())

        for j, sigma in enumerate(gaussian_sigmas):
            noise = torch.randn_like(batch_images) * sigma
            noisy_images = torch.clamp(batch_images + noise, -1000, 1000)
            noisy_output = model(noisy_images)
            noisy_loss = criterion(noisy_output, batch_labels)
            all_gaussian_losses[j].append(noisy_loss.item())

    fgsm_avg = [np.mean(loss_list) for loss_list in all_fgsm_losses]
    gauss_avg = [np.mean(loss_list) for loss_list in all_gaussian_losses]

    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.plot(range(0,10), fgsm_avg, marker='o')
    plt.title("FGSM Loss vs. eps=0.5...5/255")
    plt.xlabel("eps (integer, /511)")
    plt.ylabel("CE Loss")

    plt.subplot(1,2,2)
    plt.plot([0.1*i for i in range(0,10)], gauss_avg, marker='o')
    plt.title("Gaussian Loss vs. sigma=0.2...2")
    plt.xlabel("sigma")
    plt.ylabel("CE Loss")
    plt.tight_layout()
    plt.show()

    fgsm_all_flat = np.concatenate(all_fgsm_losses, axis=0)
    gauss_all_flat = np.concatenate(all_gaussian_losses, axis=0)

    plt.figure(figsize=(6,5))
    plt.hist2d(gauss_all_flat, fgsm_all_flat, bins=60)
    plt.colorbar(label='Count')
    plt.xlabel('Gaussian CE Loss (all sigma combined)')
    plt.ylabel('FGSM CE Loss (all eps combined)')
    plt.title('2D Histogram of All Losses')
    plt.show()

if __name__ == "__main__":
    main()
