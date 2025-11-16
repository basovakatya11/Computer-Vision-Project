import os
import xml.etree.ElementTree as ET
import cv2
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Parse XML annotations
def parse_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    objects = []
    for obj in root.findall("object"):
        class_name = obj.find("name").text
        if class_name == "mask_weared_incorrect":
            continue
        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)
        objects.append({
            "class": "Mask" if class_name == "with_mask" else "No_Mask",
            "bbox": [xmin, ymin, xmax, ymax]
        })
    return objects

# IOU function
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

# Check if prediction matches GT
def is_matched(pred_box, gt_box, iou_threshold=0.2):
    # Case 1: GT fully inside predicted box
    if (pred_box[0] <= gt_box[0] <= gt_box[2] <= pred_box[2] and
        pred_box[1] <= gt_box[1] <= gt_box[3] <= pred_box[3]):
        return True
    # Case 2: IoU >= threshold
    return iou(pred_box, gt_box) >= iou_threshold

# Validate system
def validate_system(image_dir, xml_dir, yolo_model, resnet_model, transform, device='cpu'):
    y_true = []
    y_pred = []

    y_true_yolo = []
    y_pred_yolo = []

    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]

    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        xml_path = os.path.join(xml_dir, os.path.splitext(img_file)[0]+'.xml')
        if not os.path.exists(xml_path):
            continue

        gt_objects = parse_annotation(xml_path)
        if len(gt_objects) == 0:
            continue

        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # YOLO detection
        results = yolo_model.predict(img_rgb)
        yolo_boxes = []
        for r in results:
            for box in r.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                yolo_boxes.append([x1, y1, x2, y2])

        # YOLO evaluation
        for gt in gt_objects:
            matched = any(is_matched(pred_box, gt['bbox']) for pred_box in yolo_boxes)
            y_true_yolo.append("Person")
            y_pred_yolo.append("Person" if matched else "No_Pred")

        # Full system evaluation
        for gt in gt_objects:
            best_box = None
            for pred_box in yolo_boxes:
                if is_matched(pred_box, gt['bbox']):
                    best_box = pred_box
                    break

            if best_box is None:
                continue

            x1, y1, x2, y2 = best_box
            # Crop upper half or upper third
            if (y2 - y1) > 0.6 * img.shape[0]:
                face_crop = img_rgb[y1:y2, x1:x2]
            else:
                y_face2 = y1 + (y2 - y1) // 2
                face_crop = img_rgb[y1:y_face2, x1:x2]

            if face_crop.size == 0:
                continue

            crop_resized = cv2.resize(face_crop, (224, 224))
            crop_tensor = transform(crop_resized).unsqueeze(0).to(device)

            with torch.no_grad():
                output = resnet_model(crop_tensor)
                pred_class_idx = torch.argmax(output, dim=1).item()
                pred_class = "Mask" if pred_class_idx == 0 else "No_Mask"

            y_true.append(gt['class'])
            y_pred.append(pred_class)

    # Metrics
    precision_yolo = precision_score(y_true_yolo, y_pred_yolo, pos_label="Person", zero_division=0)
    recall_yolo = recall_score(y_true_yolo, y_pred_yolo, pos_label="Person", zero_division=0)
    f1_yolo = f1_score(y_true_yolo, y_pred_yolo, pos_label="Person", zero_division=0)

    labels_sys = ["Mask", "No_Mask", "No_Pred"]
    precision_sys = precision_score(y_true, y_pred, labels=labels_sys, average='macro', zero_division=0)
    recall_sys = recall_score(y_true, y_pred, labels=labels_sys, average='macro', zero_division=0)
    f1_sys = f1_score(y_true, y_pred, labels=labels_sys, average='macro', zero_division=0)

    print(f"YOLO detection - Precision: {precision_yolo:.3f}, Recall: {recall_yolo:.3f}, F1: {f1_yolo:.3f}")
    print(f"Full system  - Precision: {precision_sys:.3f}, Recall: {recall_sys:.3f}, F1: {f1_sys:.3f}")

    # 5. Confusion matrices
    cm_yolo = confusion_matrix(y_true_yolo, y_pred_yolo, labels=["Person", "No_Pred"])
    cm_sys = confusion_matrix(y_true, y_pred, labels=["Mask", "No_Mask"])

    # Save as CSV
    pd.DataFrame(cm_yolo, index=["True_Person", "True_No_Pred"], columns=["Pred_Person", "Pred_No_Pred"]).to_csv("yolo_confusion_matrix.csv")
    pd.DataFrame(cm_sys, index=["True_Mask", "True_No_Mask"], columns=["Pred_Mask", "Pred_No_Mask"]).to_csv("system_confusion_matrix.csv")
    print("[INFO] Confusion matrices saved as CSV files.")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.heatmap(cm_yolo, annot=True, fmt='d', cmap='Blues', ax=axes[0], xticklabels=["Pred_Person", "Pred_No_Pred"], yticklabels=["True_Person", "True_No_Pred"])
    axes[0].set_title("YOLO Confusion Matrix")
    sns.heatmap(cm_sys, annot=True, fmt='d', cmap='Greens', ax=axes[1], xticklabels=["Pred_Mask", "Pred_No_Mask"], yticklabels=["True_Mask", "True_No_Mask"])
    axes[1].set_title("Full System (ResNet) Confusion Matrix")
    plt.tight_layout()
    plt.show()
