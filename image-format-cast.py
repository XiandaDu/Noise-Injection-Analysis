import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

class ResNet18FeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.resnet18 = models.resnet18(pretrained=pretrained)

    def forward(self, x):
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)

        out_layer1 = self.resnet18.layer1(x)
        out_layer2 = self.resnet18.layer2(out_layer1)
        out_layer3 = self.resnet18.layer3(out_layer2)
        out_layer4 = self.resnet18.layer4(out_layer3)
        
        return out_layer1, out_layer2, out_layer3, out_layer4

def main():
    val_dir = "./Domain-Specific-Dataset/val"
    save_dir = "./Domain-Specific-Dataset/pt"
    os.makedirs(save_dir, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    feature_extractor = ResNet18FeatureExtractor(pretrained=True)
    feature_extractor.eval()
    
    for label_name in os.listdir(val_dir):
        label_path = os.path.join(val_dir, label_name)
        if not os.path.isdir(label_path):
            continue
        
        image_names = []
        layer1_feats = []
        layer2_feats = []
        layer3_feats = []
        layer4_feats = []

        for img_file in os.listdir(label_path):
            if not (img_file.lower().endswith('.jpg') or
                    img_file.lower().endswith('.jpeg') or
                    img_file.lower().endswith('.png')):
                continue
            
            img_path = os.path.join(label_path, img_file)
            with Image.open(img_path).convert('RGB') as img:
                img_tensor = transform(img)
            
            img_tensor = img_tensor.unsqueeze(0)
            
            with torch.no_grad():
                f1, f2, f3, f4 = feature_extractor(img_tensor)
            
            f1 = f1.squeeze(0)
            f2 = f2.squeeze(0)
            f3 = f3.squeeze(0)
            f4 = f4.squeeze(0)
            
            layer1_feats.append(f1.cpu())
            layer2_feats.append(f2.cpu())
            layer3_feats.append(f3.cpu())
            layer4_feats.append(f4.cpu())
            image_names.append(img_file)

        if len(layer1_feats) == 0:
            continue

        layer1_tensor = torch.stack(layer1_feats)
        layer2_tensor = torch.stack(layer2_feats)
        layer3_tensor = torch.stack(layer3_feats)
        layer4_tensor = torch.stack(layer4_feats)
        
        torch.save({
            'label': label_name,
            'image_names': image_names,
            'features': layer1_tensor
        }, os.path.join(save_dir, f"{label_name}-layer1.pt"))
        
        torch.save({
            'label': label_name,
            'image_names': image_names,
            'features': layer2_tensor
        }, os.path.join(save_dir, f"{label_name}-layer2.pt"))
        
        torch.save({
            'label': label_name,
            'image_names': image_names,
            'features': layer3_tensor
        }, os.path.join(save_dir, f"{label_name}-layer3.pt"))
        
        torch.save({
            'label': label_name,
            'image_names': image_names,
            'features': layer4_tensor
        }, os.path.join(save_dir, f"{label_name}-layer4.pt"))

        print(f"Saved {label_name} 4 layer feature maps {save_dir}")

if __name__ == "__main__":
    main()
