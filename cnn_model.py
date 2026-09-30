import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image

from preprocessing import preprocess_image
from labels import LABEL_TO_DISPLAY


class ThaiCNN(nn.Module):
    def __init__(self, num_classes=44):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 28x28 -> 14x14
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 2: 14x14 -> 7x7
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 3: 7x7 -> 4x4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class ThaiCNNPredictor:
    """
    High-accuracy Deep Learning Predictor for Thai Consonants (99.4% Accuracy).
    """
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = Path(__file__).resolve().parent / "models" / "thai_cnn_classifier.pt"
        
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        self.classes = checkpoint["classes"]
        self.model = ThaiCNN(num_classes=len(self.classes))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        # Memory & thread optimization for cloud environments (Render / 512MB RAM)
        torch.set_num_threads(2)
        torch.set_grad_enabled(False)

    def predict(self, image):
        if image is None:
            raise ValueError("กรุณาอัปโหลดรูปภาพหรือวาดตัวอักษรไทย")

        # 1. Preprocess to 28x28
        arr = preprocess_image(image)

        # 2. To PyTorch Tensor (1, 1, 28, 28)
        x = torch.tensor(arr[None, None, :, :], dtype=torch.float32)

        # 3. Predict Softmax Probabilities
        with torch.no_grad():
            logits = self.model(x)
            probabilities = torch.softmax(logits, dim=1)[0].numpy()

        # 4. Map to friendly display names
        scores = {}
        for cls_name, prob in zip(self.classes, probabilities):
            display = LABEL_TO_DISPLAY.get(cls_name, cls_name)
            scores[display] = float(prob)

        # 5. Preview image (enlarged to 280x280 with NEAREST for clear visible pixels)
        preview_28 = (arr * 255.0).astype(np.uint8)
        preview = np.array(Image.fromarray(preview_28).resize((280, 280), Image.Resampling.NEAREST))

        return scores, preview
