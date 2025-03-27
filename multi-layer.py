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
    """Extract features from ResNet18's layer1 and layer4."""
    def __init__(self, original_model):
        super(FeatureExtractorModified, self).__init__()
        self.original_model = original_model

    def forward(self, x):
        x2 = self.original_model.layer2(x)
        x3 = self.original_model.layer3(x2)
        x4 = self.original_model.layer4(x3)
        return x2

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
        return x1, x2

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

from torchattacks.attack import Attack


class FGSM(Attack):
    r"""
    FGSM in the paper 'Explaining and harnessing adversarial examples'
    [https://arxiv.org/abs/1412.6572]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.FGSM(model, eps=8/255)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, eps=8 / 255):
        super().__init__("FGSM", model)
        self.eps = eps
        self.supported_mode = ["default", "targeted"]

    def forward(self, images, labels, max_mag):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()

        images.requires_grad = True
        outputs = self.get_logits(images)

        # Calculate loss
        if self.targeted:
            cost = -loss(outputs, target_labels)
        else:
            cost = loss(outputs, labels)

        # Update adversarial images
        grad = torch.autograd.grad(
            cost, images, retain_graph=False, create_graph=False
        )[0]

        adv_images = images + self.eps * grad.sign()
        adv_images = torch.clamp(adv_images, min=0, max=max_mag).detach()

        return adv_images


def main():
    SAMPLES = 2
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
        attacker.set_normalization_used(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        fgsm_attacks.append(attacker)

    feature_extractor = FeatureExtractor(model.to(device))
    feature_extractor.eval()

    feature_extractor_modified = FeatureExtractorModified(model_modified.to(device))
    feature_extractor_modified.eval()

    for i in range(0,10):
        fgsm_attacker = fgsm_attacks[i]

        fgsm_loss_list = []
        eps_grid = np.arange(0, 10, 1)
        layer2_loss_sum = {e: 0.0 for e in eps_grid}
        layer2_loss_count = {e: 0 for e in eps_grid}

        for batch_images, batch_labels in dataloader:
            batch_images = batch_images.to(device)
            batch_labels = torch.tensor([int(lbl) for lbl in batch_labels], dtype=torch.long).to(device)

            with torch.no_grad():
                f1_orig, _ = feature_extractor(batch_images)
            f1_orig.requires_grad = True

            adv_images = fgsm_attacker(batch_images, batch_labels)
            fgsm_output = model(adv_images)
            fgsm_loss = criterion(fgsm_output, batch_labels)
            fgsm_loss_list.append(fgsm_loss.item())

            fixed_input = f1_orig.detach()
            fixed_input.requires_grad = False

            for e2 in eps_grid:
                attacker_layer2 = FGSM(model_modified, eps=e2/255.0)
                try:
                    max_mag = fixed_input.max()
                    modified_layer2 = attacker_layer2.forward(fixed_input, batch_labels, max_mag)
                    output_layer2 = model_modified(modified_layer2)
                    loss_layer2 = criterion(output_layer2, batch_labels)
                    layer2_loss_sum[e2] += loss_layer2.item()
                    layer2_loss_count[e2] += 1
                except Exception as e:
                    print(f"An error occurred: {e}")
                    continue

        eps_list = []
        loss_avg_list = []
        for e in eps_grid:
            if layer2_loss_count[e] > 0:
                eps_list.append(e)
                loss_avg_list.append(layer2_loss_sum[e] / layer2_loss_count[e])

        fgsm_loss_avg = sum(fgsm_loss_list) / len(fgsm_loss_list)

        plt.figure(figsize=(7,5))
        plt.plot(eps_list, loss_avg_list, color='blue', label='Layer2 epsilon sweep (avg)')
        plt.axhline(y=fgsm_loss_avg, color='red', linestyle='--', label=f'Layer1 FGSM loss={fgsm_loss_avg:.4f}')
        plt.xlabel("Layer2 epsilon")
        plt.ylabel("Cross Entropy Loss")
        plt.title(f"Avg Loss Curve | Layer1 eps={0.5*i/255:.4f}")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"./2D-hist/loss_avg_curve_layer1eps_{0.5*i:.1f}.png", dpi=200)
        plt.close()


        # orig_l1, orig_l4 = gather_and_random_downsample(orig_layer1_list, orig_layer2_list)
        # fgsm_l1, fgsm_l4 = gather_and_random_downsample(fgsm_layer1_list, fgsm_layer2_list)
        # modified_l1, modified_l4 = gather_and_random_downsample(fgsm_layer1_list, modified_layer2_list)

        # fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        # fig.suptitle(f"FGSM eps={i/2:.1f}/255, Modified epsilon", fontsize=14)

        # h1 = axs[0].hist2d(orig_l1, orig_l4, bins=50, range=[[0,3],[0,3]], cmap='viridis')
        # axs[0].set_title("Original (layer1 vs. layer4)")
        # axs[0].set_xlabel("Layer1 feature")
        # axs[0].set_ylabel("Layer4 feature")
        # fig.colorbar(h1[3], ax=axs[0])

        # h2 = axs[1].hist2d(fgsm_l1, fgsm_l4, bins=50, range=[[0,3],[0,3]], cmap='viridis')
        # axs[1].set_title("FGSM (layer1 vs. layer4)")
        # axs[1].set_xlabel("Layer1 feature")
        # axs[1].set_ylabel("Layer4 feature")
        # fig.colorbar(h2[3], ax=axs[1])

        # h3 = axs[2].hist2d(modified_l1, modified_l4, bins=50, range=[[0,3],[0,3]], cmap='viridis')
        # axs[2].set_title("Modified (layer1 vs. layer4)")
        # axs[2].set_xlabel("Layer1 feature")
        # axs[2].set_ylabel("Layer4 feature")
        # fig.colorbar(h3[3], ax=axs[2])

        # plt.tight_layout()
        # plt.subplots_adjust(bottom=0.2)

        # out_filename = f"./2D-hist/2Dhist_eps_{i/2:.1f}.png"
        # plt.savefig(out_filename, dpi=200)
        # plt.close(fig)



if __name__ == "__main__":
    main()
