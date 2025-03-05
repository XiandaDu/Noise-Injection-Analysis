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
        """随机选择指定数量的样本。"""
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
    """从 ResNet18 中提取 layer1 和 layer4 的特征。"""
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
        return x1, x4   # 返回 layer1, layer4 即可

def gather_and_random_downsample(list_f1, list_f4):
    """
    将多个 batch 的 (layer1, layer4) flatten 后合并，并通过随机下采样
    使二者长度一致，再返回 (f1_sample, f4_sample)。
    """
    # 拼接为大的一维数组
    f1_array = np.concatenate(list_f1, axis=0)  # shape ~ (N_total, )
    f4_array = np.concatenate(list_f4, axis=0)  # shape ~ (M_total, )

    # (可选) 过滤掉 =0 的值
    mask_f1 = (f1_array != 0)
    mask_f4 = (f4_array != 0)
    f1_nonzero = f1_array[mask_f1]
    f4_nonzero = f4_array[mask_f4]

    # 随机下采样到相同长度
    min_len = min(len(f1_nonzero), len(f4_nonzero))
    if min_len == 0:
        # 如果全是 0，就返回空
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

    # 准备若干 FGSM 攻击（这里 eps 从 0 到 0.5, 步长 0.5/9）
    fgsm_attacks = []
    for e in range(10):
        attacker = torchattacks.FGSM(model, eps=0.5*e/255.0)
        attacker.normalization_used = True
        attacker.set_normalization_used(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        fgsm_attacks.append(attacker)

    # 准备若干 Gaussian 噪声 sigma (0, 0.2, 0.4, ..., 1.8)
    gaussian_sigmas = [0.2 * i for i in range(10)]

    # 这里我们选「最大 eps」和「最大 sigma」来对比，可自行改成别的
    max_fgsm_attacker = fgsm_attacks[9]      # eps = 0.5*9 / 255
    max_gaussian_sigma = gaussian_sigmas[9]  # 0.2*9 = 1.8

    feature_extractor = FeatureExtractor(model.to(device))
    feature_extractor.eval()

    # 用来收集(Original, FGSM, Gaussian)的 layer1, layer4 特征
    orig_layer1_list,  orig_layer4_list  = [], []
    fgsm_layer1_list,  fgsm_layer4_list  = [], []
    gauss_layer1_list, gauss_layer4_list = [], []

    # 遍历数据集，每张图像分别获取: 原图、FGSM、高斯噪声图 的 (layer1, layer4)
    for batch_images, batch_labels in dataloader:
        batch_images = batch_images.to(device)
        batch_labels = torch.tensor([int(lbl) for lbl in batch_labels], dtype=torch.long).to(device)

        with torch.no_grad():
            f1_orig, f4_orig = feature_extractor(batch_images)
        orig_layer1_list.append(f1_orig.view(-1).cpu().numpy())
        orig_layer4_list.append(f4_orig.view(-1).cpu().numpy())

        # FGSM 对抗图 (最大 eps)
        adv_images = max_fgsm_attacker(batch_images, batch_labels)
        with torch.no_grad():
            f1_fgsm, f4_fgsm = feature_extractor(adv_images)
        fgsm_layer1_list.append(f1_fgsm.view(-1).cpu().numpy())
        fgsm_layer4_list.append(f4_fgsm.view(-1).cpu().numpy())

        # Gaussian 噪声图 (最大 sigma)
        noise = torch.randn_like(batch_images) * max_gaussian_sigma
        gauss_images = torch.clamp(batch_images + noise, -1000, 1000)
        with torch.no_grad():
            f1_gauss, f4_gauss = feature_extractor(gauss_images)
        gauss_layer1_list.append(f1_gauss.view(-1).cpu().numpy())
        gauss_layer4_list.append(f4_gauss.view(-1).cpu().numpy())

    # 下采样到相同长度，得到可以对应绘制的 (x, y)
    orig_l1,  orig_l4  = gather_and_random_downsample(orig_layer1_list,  orig_layer4_list)
    fgsm_l1,  fgsm_l4  = gather_and_random_downsample(fgsm_layer1_list,  fgsm_layer4_list)
    gauss_l1, gauss_l4 = gather_and_random_downsample(gauss_layer1_list, gauss_layer4_list)

    # 画图 (1×3)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Original
    h1 = axs[0].hist2d(orig_l1, orig_l4, bins=50, range=[[0,3],[0,3]], cmap='viridis')
    axs[0].set_title("Original (layer1 vs. layer4)")
    axs[0].set_xlabel("Layer1 feature value")
    axs[0].set_ylabel("Layer4 feature value")
    fig.colorbar(h1[3], ax=axs[0])

    # FGSM
    h2 = axs[1].hist2d(fgsm_l1, fgsm_l4, bins=50, range=[[0,3],[0,3]], cmap='viridis')
    axs[1].set_title("FGSM (layer1 vs. layer4)")
    axs[1].set_xlabel("Layer1 feature value")
    axs[1].set_ylabel("Layer4 feature value")
    fig.colorbar(h2[3], ax=axs[1])

    # Gaussian
    h3 = axs[2].hist2d(gauss_l1, gauss_l4, bins=50, range=[[0,3],[0,3]], cmap='viridis')
    axs[2].set_title("Gaussian (layer1 vs. layer4)")
    axs[2].set_xlabel("Layer1 feature value")
    axs[2].set_ylabel("Layer4 feature value")
    fig.colorbar(h3[3], ax=axs[2])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
