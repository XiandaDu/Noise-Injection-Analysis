import os
import torch
from PIL import Image
import torchvision.transforms as transforms

def main():
    val_dir = "./Domain-Specific-Dataset/val"
    save_dir = "./Domain-Specific-Dataset/pt"

    os.makedirs(save_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)), # resize to 224x224
        transforms.ToTensor()
    ])

    for label_name in os.listdir(val_dir):
        label_path = os.path.join(val_dir, label_name)
        if not os.path.isdir(label_path):
            continue

        image_tensors = []
        image_names = []

        for img_file in os.listdir(label_path):
            img_path = os.path.join(label_path, img_file)
            if not img_file.lower().endswith('.jpeg'):
                continue

            with Image.open(img_path).convert('RGB') as img:
                img_tensor = transform(img)

            image_tensors.append(img_tensor)
            image_names.append(img_file)

        if len(image_tensors) == 0:
            continue

        data_dict = {
            'label': label_name,
            'image_names': image_names,
            'tensors': torch.stack(image_tensors)  # (N, 3, 224, 224)
        }

        save_path = os.path.join(save_dir, f"{label_name}.pt")
        torch.save(data_dict, save_path)
        print(f"Saved {label_name} to {save_path}")

if __name__ == "__main__":
    main()
