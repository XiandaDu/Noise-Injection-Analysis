import os 
import random
import math
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import numpy as np

##################################
# 1. Custom Dataset
##################################
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
        # Convert label_name back to class index
        class_index = self.class_to_idx[label_name]
        return sample, class_index


###########################################
# 2. Forward hook to capture layer output
###########################################
class SaveFeatures:
    """
    Captures the forward output and stores the gradient.
    """
    def __init__(self):
        self.output = None
        self.grad = None

    def hook_fn(self, module, input, output):
        # Force requires_grad=True for backward pass
        self.output = output.requires_grad_()

        # Hook to grab gradient
        def hook_grad(grad):
            self.grad = grad
        self.output.register_hook(hook_grad)


###################################
# 3. FGSM at an intermediate layer
###################################
def fgsm_inject_at_layer(model, x, y, layer_module, epsilon, device):
    """
    Add FGSM noise to layer output and measure the adversarial loss.
    """
    hook_saver = SaveFeatures()
    hook_handle = layer_module.register_forward_hook(hook_saver.hook_fn)

    model.zero_grad()

    # Forward pass
    logits = model(x)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, y)

    # Backprop to get gradient
    loss.backward(retain_graph=True)

    grad_sign = hook_saver.grad.sign()
    output_adv = hook_saver.output + epsilon * grad_sign

    # Replace forward function with patched one
    original_forward = layer_module.forward
    def patched_forward(*args):
        return output_adv
    layer_module.forward = patched_forward

    # Second pass to get new logits & CE
    logits_adv = model(x)
    loss_adv = criterion(logits_adv, y)

    # Restore
    layer_module.forward = original_forward
    hook_handle.remove()

    return logits_adv, loss_adv.item()


#######################################################
# 4. Match cross-entropy: layer2's epsilon to layer1's
#######################################################
def match_ce_between_layers(model, x, y, layer1, layer2, epsilon_layer1, device):
    """
    Binary search for epsilon_layer2 that matches CE from layer1.
    """
    # Step 1: get target CE from layer1
    _, ce_layer1 = fgsm_inject_at_layer(model, x, y, layer1, epsilon_layer1, device)

    # Step 2: match CE at layer2 using binary search
    target_ce = ce_layer1
    epsilon_min, epsilon_max = 0.0, 1.0
    best_epsilon = 0.0
    best_ce = float('inf')

    for _ in range(15):  # ~15 steps
        mid = (epsilon_min + epsilon_max) / 2.0
        _, ce_layer2 = fgsm_inject_at_layer(model, x, y, layer2, mid, device)

        if ce_layer2 < target_ce:
            epsilon_min = mid
        else:
            epsilon_max = mid

        if abs(ce_layer2 - target_ce) < abs(best_ce - target_ce):
            best_ce = ce_layer2
            best_epsilon = mid

    return ce_layer1, best_epsilon, best_ce


######################
# 5. Main
######################
def main():
    random.seed(0)
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pretrained resnet18
    model = models.resnet18(pretrained=True).eval().to(device)

    # Reference layers for hooking
    layer1 = model.layer1
    layer2 = model.layer2

    # Example transform for ImageNet
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    # Create dataset & dataloader (sample 2 images/class)
    dataset_dir = "./Domain-Specific-Dataset/val"
    dataset = CustomImageFolder(root=dataset_dir, transform=transform, samples_per_class=2)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Fixed epsilon for layer1
    epsilon_layer1 = 0.01

    # Loop over images
    for i, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        ce1, found_eps2, ce2 = match_ce_between_layers(
            model, images, labels, layer1, layer2, epsilon_layer1, device
        )

        print(f"Image {i}:")
        print(f"  FGSM injection at layer1 with eps={epsilon_layer1:.4f} => CE = {ce1:.4f}")
        print(f"  Found eps={found_eps2:.4f} at layer2 => CE = {ce2:.4f}")

        # Limit to a few images
        if i >= 10:
            break


if __name__ == "__main__":
    main()
