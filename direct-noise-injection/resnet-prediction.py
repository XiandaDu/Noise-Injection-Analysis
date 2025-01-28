import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image

model = resnet18(pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

image_path = "../original_5.png"
image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    output = model(input_tensor)

probabilities = torch.nn.functional.softmax(output[0], dim=0)
top5_prob, top5_catid = torch.topk(probabilities, 5)  # Get Top 5 Results

imagenet_classes = {idx: entry.strip() for idx, entry in enumerate(open("imagenet_classes.txt"))}

print("Top-5 Predictions:")
for i in range(5):
    print(f"{imagenet_classes[top5_catid[i].item()]}: {top5_prob[i].item():.4f}")
