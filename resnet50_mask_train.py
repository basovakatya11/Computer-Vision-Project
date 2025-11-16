import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import precision_score, recall_score, f1_score
import time
import copy
import csv
import os

# PARAMETERS
DATA_DIR = "datasets"
BATCH_SIZE = 32
NUM_CLASSES = 2
EPOCHS = 5
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TRANSFORMS AND DATA
train_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_ds = ImageFolder(f"{DATA_DIR}/train", transform=train_tfms)
val_ds = ImageFolder(f"{DATA_DIR}/val", transform=val_tfms)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)

# MODEL
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# LOGGING SETUP
log_path = "metrics_log.csv"
if not os.path.exists(log_path):
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Class", "Precision", "Recall", "F1"])

# TRAINING LOOP
best_acc = 0.0
best_model_wts = copy.deepcopy(model.state_dict())

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print('-' * 30)

    #  TRAINING PHASE
    model.train()
    running_loss, running_corrects = 0.0, 0

    for imgs, labels in train_dl:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * imgs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(train_ds)
    epoch_acc = running_corrects.double() / len(train_ds)
    print(f"Train Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")

    #  VALIDATION PHASE
    model.eval()
    val_corrects = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in val_dl:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)

            val_corrects += torch.sum(preds == labels.data)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_acc = val_corrects.double() / len(val_ds)

    # Compute validation metrics
    precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
    recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
    f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)

    print(f"Val Acc: {val_acc:.4f}")
    print(f"Precision (Mask, No_Mask): {precision}")
    print(f"Recall (Mask, No_Mask): {recall}")
    print(f"F1 (Mask, No_Mask): {f1}")

    # Save metrics to CSV
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        for i, cls in enumerate(["Mask", "No_Mask"]):
            writer.writerow([epoch + 1, cls, precision[i], recall[i], f1[i]])

    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(best_model_wts, "resnet50_mask_best.pth")
        print("✅ Model saved (best so far).")

print(f"\nTraining complete. Best val acc: {best_acc:.4f}")
