import os 
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm

def save_features_to_csv(features_dict, file_name):
    columns = []
    layer_names = ["Layer_1", "Layer_2", "Layer_3", "Layer_4"]  # New column names
    
    for noise_level, data in features_dict.items():
        for idx, (key, values) in enumerate(data.items()):
            if idx < len(layer_names):  # Ensure indexing within the defined names
                columns.append(pd.Series(values, name=f"{layer_names[idx]}"))
    
    df = pd.concat(columns, axis=1)
    df.to_csv(file_name, index=False)
    print(f"Saved features CSV: {file_name}")

def save_predictions_to_csv(predictions_dict, file_name):
    columns = []
    for key, values in predictions_dict.items():
        columns.append(pd.Series(values, name=key))
    df = pd.concat(columns, axis=1)
    df.to_csv(file_name, index=False)
    print(f"Saved predictions CSV: {file_name}")

def main():
    model = models.resnet18(pretrained=True)
    model.eval()
    
    features = {}
    def hook(module, input, output):
        features[module] = output
    
    layers_to_hook = ["layer1", "layer2", "layer3", "layer4"]
    for name, layer in model.named_modules():
        if name in layers_to_hook:
            layer.register_forward_hook(hook)
    
    with open("imagenet_classes.txt", "r") as f:
        imagenet_classes = [line.strip() for line in f.readlines()]
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    def inject_noise(image_tensor, noise_level):
        noise = torch.randn_like(image_tensor) * noise_level
        return image_tensor + noise
    
    def save_image_from_tensor(image_tensor, file_name):
        unnormalize = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        )
        unnormalized_tensor = unnormalize(image_tensor.squeeze(0))
        clipped_tensor = torch.clamp(unnormalized_tensor, 0, 1)
        img = transforms.ToPILImage()(clipped_tensor)
        img.save(file_name)
        print(f"Saved image: {file_name}")
    
    dataset_root = "../Domain-Specific-Dataset/val"
    output_root = "./gaussian_results"
    os.makedirs(output_root, exist_ok=True)
    
    noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    for label in os.listdir(dataset_root):
        label_path = os.path.join(dataset_root, label)
        if not os.path.isdir(label_path):
            continue
        
        print(f"\nProcessing label: {label}")
        output_label_dir = os.path.join(output_root, label)
        os.makedirs(output_label_dir, exist_ok=True)
        
        image_files = [f for f in os.listdir(label_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:40]
        
        for idx, image_file in enumerate(tqdm(image_files, desc=f"Processing images for {label}")):
            image_path = os.path.join(label_path, image_file)
            image_output_dir = os.path.join(output_label_dir, f"image{idx+1}")
            os.makedirs(image_output_dir, exist_ok=True)
            
            image = Image.open(image_path).convert("RGB")
            input_tensor = transform(image).unsqueeze(0)
            
            features_dict = {}
            features.clear()
            with torch.no_grad():
                output = model(input_tensor)
            
            original_features = {name: features[layer].view(-1).cpu().numpy()
                                 for name, layer in model.named_modules()
                                 if name in layers_to_hook and layer in features}
            features_dict["original"] = original_features
            
            probabilities = torch.nn.functional.softmax(output, dim=1)[0]
            top10_prob, top10_catid = torch.topk(probabilities, 10)
            classifications = [imagenet_classes[catid] for catid in top10_catid]
            
            original_predictions = {
                "classification": classifications,
                "probability": top10_prob.cpu().numpy()
            }
            csv_orig_pred = os.path.join(image_output_dir, f"predictions_nl_0.csv")
            save_predictions_to_csv(original_predictions, csv_orig_pred)
            
            orig_image_path = os.path.join(image_output_dir, f"original_{idx+1}_nl_0.jpeg")
            save_image_from_tensor(input_tensor, orig_image_path)
            csv_orig = os.path.join(image_output_dir, f"original_features_{idx+1}_nl_0.csv")
            save_features_to_csv({"original": original_features}, csv_orig)
            
            for nl in noise_levels:
                noisy_input = inject_noise(input_tensor, nl)
                noisy_image_path = os.path.join(image_output_dir, f"gaussian_{idx+1}_nl_{nl}.jpeg")
                save_image_from_tensor(noisy_input, noisy_image_path)
                
                features.clear()
                with torch.no_grad():
                    output = model(noisy_input)
                
                noisy_features = {name: features[layer].view(-1).cpu().numpy()
                                  for name, layer in model.named_modules()
                                  if name in layers_to_hook and layer in features}
                
                probabilities = torch.nn.functional.softmax(output, dim=1)[0]
                top10_prob, top10_catid = torch.topk(probabilities, 10)
                classifications = [imagenet_classes[catid] for catid in top10_catid]
                
                predictions = {
                    "classification": classifications,
                    "probability": top10_prob.cpu().numpy()
                }
                csv_pred = os.path.join(image_output_dir, f"predictions_nl_{nl}.csv")
                save_predictions_to_csv(predictions, csv_pred)
                
                csv_noisy = os.path.join(image_output_dir, f"gaussian_features_{idx+1}_nl_{nl}.csv")
                save_features_to_csv({f"": noisy_features}, csv_noisy)

if __name__ == "__main__":
    main()
