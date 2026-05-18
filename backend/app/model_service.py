from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms


ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT_DIR / "models" / "best_mobilenetv2.pth"
IMAGE_SIZE = 224


class ModelNotReadyError(RuntimeError):
    """Raised when the trained model file is not available yet."""


class InvalidImageError(ValueError):
    """Raised when the uploaded file is not a readable image."""


class TomatoDiseasePredictor:
    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: torch.nn.Module | None = None
        self.classes: list[str] = []
        self.transform = transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    @property
    def is_ready(self) -> bool:
        return self.model_path.exists()

    def status(self) -> dict[str, Any]:
        return {
            "model_ready": self.is_ready,
            "model_path": str(self.model_path),
            "device": str(self.device),
            "classes": self.classes,
        }

    def _build_model(self, num_classes: int) -> torch.nn.Module:
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    def load(self) -> None:
        if self.model is not None:
            return

        if not self.model_path.exists():
            raise ModelNotReadyError(
                "Model file not found. Train the model first so "
                "models/best_mobilenetv2.pth is created."
            )

        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.classes = checkpoint.get("classes", [])

        if not self.classes:
            raise ModelNotReadyError("Model checkpoint does not contain class names.")

        model = self._build_model(len(self.classes))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()

        self.model = model

    def predict(self, image: Image.Image) -> dict[str, Any]:
        self.load()

        if self.model is None:
            raise ModelNotReadyError("Model could not be loaded.")

        image = image.convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            confidence, predicted_index = torch.max(probabilities, dim=0)

        predictions = [
            {
                "class_name": class_name,
                "confidence": round(float(probabilities[index]) * 100, 2),
            }
            for index, class_name in enumerate(self.classes)
        ]
        predictions.sort(key=lambda item: item["confidence"], reverse=True)

        return {
            "class_name": self.classes[int(predicted_index)],
            "confidence": round(float(confidence) * 100, 2),
            "predictions": predictions[:5],
        }


def read_image(file_bytes: bytes) -> Image.Image:
    try:
        from io import BytesIO

        return Image.open(BytesIO(file_bytes))
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError("Please upload a valid image file.") from exc
