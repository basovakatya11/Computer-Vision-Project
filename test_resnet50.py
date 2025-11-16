import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



# 1. Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32

# Transformations (must match training preprocessing)
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# 2. Load test dataset
test_dir = "datasets/test"  # change to your path
test_data = datasets.ImageFolder(test_dir, transform=test_transform)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

class_names = test_data.classes  # e.g. ['Mask', 'No_Mask']
print(f"Classes: {class_names}")


# 3. Load trained ResNet model
model = models.resnet50(weights=None)  # start from same architecture
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))
model.load_state_dict(torch.load("resnet50_mask_best.pth", map_location=device))
model = model.to(device)
model.eval()


# 4. Evaluate
y_true, y_pred = [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())


# 5. Metrics
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names, digits=3))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
cm_df.to_csv("resnet_confusion_matrix.csv")
print("[INFO] Confusion matrix saved to resnet_confusion_matrix.csv")

# Plot confusion matrix
plt.figure(figsize=(5, 4))
sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("ResNet50 Mask Classification - Confusion Matrix")
plt.show()