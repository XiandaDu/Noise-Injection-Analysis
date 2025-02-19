import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm

def save_features_to_csv_by_depth(features_dict, file_name):
    """
    以 by-depth 形式将网络层特征存储到 CSV 文件。
    features_dict 的结构示例（其中 value 是一个 [1, C, H, W] 的 Tensor）：
    {
        "layer1": <Tensor shape=[1, C1, H1, W1]>,
        "layer2": <Tensor shape=[1, C2, H2, W2]>,
        ...
    }
    存储时，会将每个 layer 的每个通道（depth）flatten 成一列。
    """
    # 建立一个映射，将模型中的 "layer1" 等名称转换为需要的列名前缀 "Layer_1" 等
    layer_map = {
        "layer1": "Layer_1",
        "layer2": "Layer_2",
        "layer3": "Layer_3",
        "layer4": "Layer_4"
    }
    
    data = {}
    max_length = 0
    
    # 遍历所有层
    for layer_name, layer_data in features_dict.items():
        # layer_data.shape = [1, channels, H, W]
        _, channels, _, _ = layer_data.shape
        
        # 根据映射获取想要的层名前缀
        prefix_name = layer_map.get(layer_name, layer_name)
        
        # 逐个通道展开
        for depth in range(channels):
            # layer_data[0, depth] shape = [H, W]
            feature_values = layer_data[0, depth].contiguous().view(-1).cpu().numpy()
            max_length = max(max_length, len(feature_values))
            
            # 拼出列名，例如 "Layer_1_depth0"
            column_name = f"{prefix_name}_depth{depth}"
            data[column_name] = feature_values
    
    # 对齐长度（可能不同层的特征展平后长度不一致）
    for key, values in data.items():
        if len(values) < max_length:
            data[key] = list(values) + [float("nan")] * (max_length - len(values))
    
    # 写入 CSV
    df = pd.DataFrame(data)
    df.to_csv(file_name, index=False)
    print(f"Saved features CSV: {file_name}")

def inject_noise(image_tensor, noise_level):
    """
    给张量添加随机高斯噪声。
    """
    noise = torch.randn_like(image_tensor) * noise_level
    return image_tensor + noise

def main():
    # 1. 定义并加载模型
    model = models.resnet18(pretrained=True)
    model.eval()
    
    # 用于获取中间层特征
    features = {}
    def hook(module, input, output):
        features[module] = output

    layers_to_hook = ["layer1", "layer2", "layer3", "layer4"]
    for name, layer in model.named_modules():
        if name in layers_to_hook:
            layer.register_forward_hook(hook)

    # 2. 定义图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    # 3. 数据集根目录、输出目录及噪声级别
    dataset_root = "../Domain-Specific-Dataset/val"
    output_root = "./gaussian_results"
    os.makedirs(output_root, exist_ok=True)
    
    noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    # 4. 遍历标签文件夹
    for label in os.listdir(dataset_root):
        label_path = os.path.join(dataset_root, label)
        if not os.path.isdir(label_path):
            continue
        
        print(f"\nProcessing label: {label}")
        output_label_dir = os.path.join(output_root, label)
        os.makedirs(output_label_dir, exist_ok=True)
        
        # 这里只读取部分图像作为示例
        image_files = [f for f in os.listdir(label_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:40]
        
        # 5. 遍历每张图像
        for idx, image_file in enumerate(tqdm(image_files, desc=f"Processing images for {label}")):
            image_path = os.path.join(label_path, image_file)
            image_output_dir = os.path.join(output_label_dir, f"image{idx+1}-by-depth")
            os.makedirs(image_output_dir, exist_ok=True)
            
            # 加载并预处理
            image = Image.open(image_path).convert("RGB")
            input_tensor = transform(image).unsqueeze(0)
            
            # 对原始图像做前向传播，提取特征
            features.clear()
            with torch.no_grad():
                _ = model(input_tensor)
            
            # 提取原始图像特征（by-depth）
            original_features = {
                name: features[layer]
                for name, layer in model.named_modules()
                if name in layers_to_hook and layer in features
            }
            
            # 保存原始特征
            csv_orig_feat = os.path.join(image_output_dir, f"original_features_{idx+1}_nl_0.csv")
            save_features_to_csv_by_depth(original_features, csv_orig_feat)
            
            # 6. 遍历每种噪声级别并保存特征
            for nl in noise_levels:
                # 加噪并前向传播
                noisy_input = inject_noise(input_tensor, nl)
                
                features.clear()
                with torch.no_grad():
                    _ = model(noisy_input)
                
                # 提取加噪后特征（by-depth）
                noisy_features = {
                    name: features[layer]
                    for name, layer in model.named_modules()
                    if name in layers_to_hook and layer in features
                }
                
                # 保存噪声特征
                csv_noisy_feat = os.path.join(image_output_dir, f"gaussian_features_{idx+1}_nl_{nl}.csv")
                save_features_to_csv_by_depth(noisy_features, csv_noisy_feat)

if __name__ == "__main__":
    main()
