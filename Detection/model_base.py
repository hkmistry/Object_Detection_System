from ultralytics import YOLO

# Load the YOLOv8 model exactly once in memory to prevent multiple processes from bloating RAM.
shared_model = YOLO("yolov8n.pt")
