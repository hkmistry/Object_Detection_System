import torch
import numpy as np
from ultralytics import YOLO

# Limit PyTorch CPU threads to prevent CPU thrashing under concurrent load
torch.set_num_threads(1)

# Load the YOLOv8 model exactly once in memory to prevent multiple processes from bloating RAM.
shared_model = YOLO("yolov8n.pt")

# Warm up the model with a dummy inference to avoid first-request latency
dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
with torch.no_grad():
    _ = shared_model(dummy_frame, verbose=False)

