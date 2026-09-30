import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

sys.path.append(str(Path(__file__).resolve().parent))
from preprocessing import preprocess_image
from labels import THAI_CONSONANTS, LABEL_TO_DISPLAY

# Ensure UTF-8
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "thai_characters"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
SAVE_PATH = MODEL_DIR / "thai_cnn_classifier.pt"

# -------------------------------------------------------------
# 1. Load Preprocessed Data
# -------------------------------------------------------------
classes = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")])
class_to_idx = {c: i for i, c in enumerate(classes)}
idx_to_class = {i: c for i, c in enumerate(classes)}

print(f"Loading {len(classes)} classes from {DATA_DIR}...")
t0 = time.time()
X, y = [], []
for c in classes:
    p = DATA_DIR / c
    for f in p.glob("*.jpg"):
        arr = preprocess_image(f)
        X.append(arr)
        y.append(class_to_idx[c])

X = np.array(X, dtype=np.float32)[:, None, :, :]  # (N, 1, 28, 28)
y = np.array(y, dtype=np.int64)
print(f"Loaded {len(X)} images in {time.time() - t0:.2f}s.")

# Split 80/20 Stratified
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Training set: {len(X_train)} samples, Test set: {len(X_test)} samples")


# -------------------------------------------------------------
# 2. PyTorch Dataset & Augmentation
# -------------------------------------------------------------
class ThaiDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = torch.tensor(images)
        self.labels = torch.tensor(labels)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


# Realistic Handwriting & Font Distortions
train_transform = T.Compose([
    T.RandomRotation(degrees=(-12, 12)),
    T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.92, 1.08)),
])

train_loader = DataLoader(
    ThaiDataset(X_train, y_train, transform=train_transform),
    batch_size=64,
    shuffle=True,
    num_workers=0,
)
test_loader = DataLoader(
    ThaiDataset(X_test, y_test),
    batch_size=128,
    shuffle=False,
    num_workers=0,
)


# -------------------------------------------------------------
# 3. Robust Thai CNN Architecture
# -------------------------------------------------------------
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


model = ThaiCNN(num_classes=len(classes))
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
epochs = 25
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# -------------------------------------------------------------
# 4. Training Loop
# -------------------------------------------------------------
print(f"\nTraining Deep CNN ({epochs} epochs on AMD Ryzen 5 5600X)...")
t_start = time.time()

for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    for bx, by in train_loader:
        optimizer.zero_grad()
        out = model(bx)
        loss = criterion(out, by)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    scheduler.step()

    if epoch % 5 == 0 or epoch == epochs:
        print(f"  Epoch [{epoch:02d}/{epochs:02d}] - Loss: {running_loss / len(train_loader):.4f}")

total_train_time = time.time() - t_start
print(f"Training finished in {total_train_time:.2f} seconds ({total_train_time / 60:.2f} mins)!")

# -------------------------------------------------------------
# 5. Comprehensive Evaluation on 2,200 Test Images
# -------------------------------------------------------------
model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for bx, by in test_loader:
        out = model(bx)
        preds = out.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(by.cpu().numpy())

all_preds = np.array(all_preds)
all_targets = np.array(all_targets)

acc = accuracy_score(all_targets, all_preds)
print("\n" + "=" * 70)
print(f"Deep CNN Test Accuracy on Combined (Handwriting + Fonts): {acc * 100:.2f}%!")
print("=" * 70)

# Save Torch model weights and metadata
torch.save({
    "model_state_dict": model.state_dict(),
    "classes": classes,
    "class_to_idx": class_to_idx,
    "accuracy": acc,
}, SAVE_PATH)

print(f"Model saved successfully to: {SAVE_PATH}")
print(f"File size: {SAVE_PATH.stat().st_size / (1024*1024):.2f} MB")
