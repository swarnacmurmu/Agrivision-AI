from pathlib import Path
from io import BytesIO

import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import models, transforms


MODEL_PATH = Path("models/best_mobilenetv2_finetuned.pth")
IMG_SIZE = 224

app = FastAPI(title="AgriVision AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    classes = checkpoint["classes"]

    model = build_model(len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, classes


model, classes = load_model()


@app.get("/")
def home():
    return {
        "message": "AgriVision AI backend is running",
        "device": str(device)
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    confidence, predicted_idx = torch.max(probabilities, dim=0)

    top_probs, top_indices = torch.topk(probabilities, k=3)

    top_predictions = []
    for prob, idx in zip(top_probs, top_indices):
        top_predictions.append({
            "class": classes[idx.item()],
            "confidence": round(prob.item() * 100, 2)
        })

    return {
        "filename": file.filename,
        "predicted_class": classes[predicted_idx.item()],
        "confidence": round(confidence.item() * 100, 2),
        "top_predictions": top_predictions
    }