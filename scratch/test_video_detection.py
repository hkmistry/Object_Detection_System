import os
import cv2
from ultralytics import YOLO

print("Loading model...")
model = YOLO("yolov8n.pt")
print("Model loaded.")

video_path = "uploads/obj_video_1.mp4"
if not os.path.exists(video_path):
    print(f"Error: {video_path} does not exist!")
    exit(1)

print(f"Opening video {video_path}...")
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Video capture could not open file.")
    exit(1)

frame_count = 0
detections_summary = {}

print("Processing frames...")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    if frame_count % 10 == 0:
        print(f"Processed {frame_count} frames...")

    results = model(frame, verbose=False)
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            name = model.names[cls]
            detections_summary[name] = detections_summary.get(name, 0) + 1

cap.release()
print(f"Finished. Total frames processed: {frame_count}")
print("Detections summary:")
for k, v in detections_summary.items():
    print(f"  {k}: {v}")
