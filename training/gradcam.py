from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


MODEL_PATH = Path("models/best_mobilenetv2_finetuned.pth")
IMAGE_PATH = Path("dataset/tomato_split/test/Tomato_Early_blight")
OUTPUT_DIR = Path("outputs/gradcam")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 224


transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def build_model(num_classes):
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        num_classes
    )
    return model


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()

        target_score = output[:, class_idx]
        target_score.backward()

        gradients = self.gradients
        activations = self.activations

        weights = gradients.mean(dim=(2, 3), keepdim=True)

        cam = (weights * activations).sum(dim=1).squeeze()
        cam = torch.relu(cam)

        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam, class_idx


def create_gradcam(image_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    classes = checkpoint["classes"]

    model = build_model(len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    target_layer = model.features[-1]

    gradcam = GradCAM(model, target_layer)

    original_image = Image.open(image_file).convert("RGB")
    input_tensor = transform(original_image).unsqueeze(0).to(device)

    cam, predicted_idx = gradcam.generate(input_tensor)

    predicted_class = classes[predicted_idx]

    image_np = np.array(original_image.resize((IMG_SIZE, IMG_SIZE)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(image_np, 0.6, heatmap, 0.4, 0)

    output_path = OUTPUT_DIR / f"gradcam_{image_file.stem}.png"
    cv2.imwrite(str(output_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    print("Predicted class:", predicted_class)
    print("Grad-CAM saved to:", output_path)


if __name__ == "__main__":
    sample_images = list(IMAGE_PATH.glob("*.JPG")) + list(IMAGE_PATH.glob("*.jpg")) + list(IMAGE_PATH.glob("*.png"))

    if not sample_images:
        print("No sample images found.")
    else:
        create_gradcam(sample_images[0])