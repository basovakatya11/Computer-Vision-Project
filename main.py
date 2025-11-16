import cv2
import numpy as np
import torch
import math
from ultralytics import YOLO
from torchvision import models, transforms
from torch import nn
from collections import OrderedDict
import time
from validate_system import validate_system

# MODEL INITIALIZATION

# YOLOv8 for person detection
yolo = YOLO('yolov8s.pt')  # use 'yolov8s.pt' for better accuracy

# ResNet-50 mask classifier
class MaskClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        base.fc = nn.Linear(base.fc.in_features, num_classes)
        self.model = base

    def forward(self, x):
        return self.model(x)

mask_model = MaskClassifier(num_classes=2)
state_dict = torch.load('resnet50_mask_best.pth', map_location='cpu')

# Remove "model." prefix if present in checkpoint
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    if k.startswith("model."):
        new_state_dict[k[6:]] = v  # remove "model." prefix
    else:
        new_state_dict[k] = v

# Load state dict safely
mask_model.load_state_dict(new_state_dict, strict=False)
mask_model.eval()

## Device setup (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mask_model.to(device)
mask_model.eval()
print(f"[INFO] Mask model loaded on device: {device}")

## Transformation pipeline for mask classifier
mask_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# DISTANCE ESTIMATION
def calibrate_f(h_pixels, H_real_m=1.70, Z_real_m=2.5):
    """Calibrate focal length (pixels) using a known person."""
    return (h_pixels * Z_real_m) / H_real_m

def estimate_distance(h_pixels, f_pixels, H_real_m=1.70):
    """Estimate distance (meters) using pinhole model."""
    return (H_real_m * f_pixels) / max(h_pixels, 1)


# PROCESS VIDEO
def process_video(input_path, output_path, f_pixels, H_real_m=1.70, distance_thresh=1.0):

    ## Read input video
    cap = cv2.VideoCapture(input_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    # Create output Video Writer
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    cx, cy = w / 2.0, h / 2.0  # approximate optical center
    print(f"[INFO] Video opened ({w}x{h}, {fps:.1f} FPS)")

    frame_count = 0
    start_time = time.time()

    # Loop through each frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Detect people in frame using YOLOv8
        results = yolo.predict(frame, conf=0.5, classes=[0], verbose=False)[0]  # class 0 = person
        boxes = results.boxes.xyxy.cpu().numpy().astype(int)

        # Initialize list of detected people - we will store their boxes and (X, Y, Z) coordinates
        people = []

        # Process each person (Bounding Box Loop)
        for (x1, y1, x2, y2) in boxes:
            h_pixels = y2 - y1
            Z = estimate_distance(h_pixels, f_pixels, H_real_m)
            x_center = (x1 + x2) / 2.0
            y_bottom = y2
            
            # Compute approximate real-world 3D coordinates
            X = (x_center - cx) * Z / f_pixels
            Y = (y_bottom - cy) * Z / f_pixels

            people.append(((x1, y1, x2, y2), (X, Y, Z)))

            # Crop upper face region for mask classification 
            y_face2 = y1 + (y2 - y1) // 2
            face_crop = frame[y1:y_face2, x1:x2]

            if face_crop.size == 0: 
                continue

            # Convert from BGR to RGB - prepare for ResNet-50 Mask Classification
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            img_t = mask_transform(face_rgb).unsqueeze(0).to(device)

            # ResNet-50 Mask Classification
            with torch.no_grad():
                pred = torch.softmax(mask_model(img_t), dim=1).cpu().numpy()[0]
            label = np.argmax(pred)

            # Draw bounding boxes and labels on frame
            label_name = ["Mask", "No_Mask"][label]
            color = (0, 255, 0) if label == 0 else (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label_name} | {Z:.2f}m", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)


        # Pairwise distance alert
        for i in range(len(people)):
            for j in range(i + 1, len(people)):
                Xi, Yi, Zi = people[i][1]
                Xj, Yj, Zj = people[j][1]
                euclid = math.sqrt((Xi - Xj)**2 + (Yi - Yj)**2 + (Zi - Zj)**2)

                if euclid < distance_thresh:
                    # Draw red line between people too close
                    p1 = (int((people[i][0][0] + people[i][0][2]) / 2),
                          int((people[i][0][1] + people[i][0][3]) / 2))
                    p2 = (int((people[j][0][0] + people[j][0][2]) / 2),
                          int((people[j][0][1] + people[j][0][3]) / 2))
                    cv2.line(frame, p1, p2, (0, 0, 255), 2)
                    cv2.putText(frame, "ALERT!", (p1[0], p1[1] - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Write the processed frame(with annotations) to output video
        out.write(frame)

    # Close the input video stream + finalize the output file + the print message
    cap.release()
    out.release()
    
    end_time = time.time()
    elapsed = end_time - start_time
    fps_processed = frame_count / elapsed if elapsed > 0 else 0.0
    print(f"[INFO] Processing complete → saved at {output_path}")
    print(f"[INFO] Processed {frame_count} frames in {elapsed:.2f} seconds ({fps_processed:.2f} FPS)")



# MAIN EXECUTION
if __name__ == "__main__":
    # Example calibration: known person height 1.7m, appears as 180px tall at 2.5m distance
    f_pixels = calibrate_f(h_pixels=180, H_real_m=1.7, Z_real_m=2.5)
    process_video("input.mp4", "output_demo.mp4", f_pixels)

    ## Optional: system evaluation(uncomment the function if you want to run it)
    #validate_system(image_dir="datasets/final/images", xml_dir="datasets/final/annotations", yolo_model=yolo, resnet_model=mask_model, transform=mask_transform, device=device)