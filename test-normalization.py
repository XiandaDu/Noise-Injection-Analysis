import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image

def save_image_from_tensor_double_inverse(image_tensor, file_name):
    unnormalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )
    unnormalized_tensor = unnormalize(image_tensor.squeeze(0))
    unnormalized_tensor = unnormalize(unnormalized_tensor.squeeze(0))
    unnormalized_tensor = unnormalize(unnormalized_tensor.squeeze(0))
    clipped_tensor = torch.clamp(unnormalized_tensor, 0, 1)
    noisy_image = transforms.ToPILImage()(clipped_tensor)
    noisy_image.save(file_name)
    print(f"Image saved to {file_name}")

def save_image_from_tensor_over_inverse(image_tensor, file_name):
    unnormalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )
    unnormalized_tensor = unnormalize(image_tensor.squeeze(0))
    unnormalized_tensor = unnormalize(unnormalized_tensor.squeeze(0))
    clipped_tensor = torch.clamp(unnormalized_tensor, 0, 1)
    noisy_image = transforms.ToPILImage()(clipped_tensor)
    noisy_image.save(file_name)
    print(f"Image saved to {file_name}")

def save_image_from_tensor(image_tensor, file_name):
    unnormalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )
    unnormalized_tensor = unnormalize(image_tensor.squeeze(0))
    clipped_tensor = torch.clamp(unnormalized_tensor, 0, 1)
    noisy_image = transforms.ToPILImage()(clipped_tensor)
    noisy_image.save(file_name)
    print(f"Image saved to {file_name}")

def save_image_from_tensor_over_norm(image_tensor, file_name):
    unnormalize = transforms.Normalize(
        mean=[0, 0, 0],
        std=[1, 1, 1]
    )
    unnormalized_tensor = unnormalize(image_tensor.squeeze(0))
    clipped_tensor = torch.clamp(unnormalized_tensor, 0, 1)
    noisy_image = transforms.ToPILImage()(clipped_tensor)
    noisy_image.save(file_name)
    print(f"Image saved to {file_name}")

def save_image_from_tensor_double_normalize(image_tensor, file_name):
    unnormalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    unnormalized_tensor = unnormalize(image_tensor.squeeze(0))
    clipped_tensor = torch.clamp(unnormalized_tensor, 0, 1)
    noisy_image = transforms.ToPILImage()(clipped_tensor)
    noisy_image.save(file_name)
    print(f"Image saved to {file_name}")

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

image_path = "./tmp.JPEG"
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

save_image_from_tensor(input_tensor, "test-original.png")
save_image_from_tensor_over_norm(input_tensor, "test-over-norm.png")
save_image_from_tensor_double_normalize(input_tensor, "test-double-norm.png")
save_image_from_tensor_over_inverse(input_tensor, "test-over-inv.png")
save_image_from_tensor_double_inverse(input_tensor, "test-double-inv.png")