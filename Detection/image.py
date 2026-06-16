import cv2
import numpy as np
from ultralytics import YOLO
from flask import request
from werkzeug.utils import secure_filename
import os

# Load YOLO model from shared base
from Detection.model_base import shared_model as model

# Folder to temporarily store uploads
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {"jpg", "jpeg", "png"}

def detect_image(file, query=None, threshold=0.25, raw=False):
    """Run YOLO detection on uploaded image and return annotated image & results"""
    if not allowed_file(file.filename):
        return None, None, []

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Read image
    img = cv2.imread(filepath)
    if img is None:
        return None, None, []
    
    import torch
    with torch.no_grad():
        results = model(img, verbose=False)

    if raw:
        # Return original image and all detections with boxes
        _, buffer = cv2.imencode('.jpg', img)
        image_bytes = buffer.tobytes()
        
        objects = []
        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            name = model.names[cls]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            objects.append({
                "name": name,
                "confidence": conf,
                "box": [x1, y1, x2, y2]
            })
        return image_bytes, img.shape, objects
    else:
        # Standard behavior
        if query:
            queries = [q.strip().lower() for q in query.split(',') if q.strip()]
        else:
            queries = []

        if queries:
            matched_indices = []
            
            # Phase 1: Try exact class matching
            for i, box in enumerate(results[0].boxes):
                cls = int(box.cls[0])
                name = model.names[cls].lower()
                conf = float(box.conf[0])
                
                exact_match = any(name == q for q in queries)
                conf_match = conf >= threshold
                if exact_match and conf_match:
                    matched_indices.append(i)
            
            # Phase 2: Substring fallback if no exact matches found
            if not matched_indices:
                for i, box in enumerate(results[0].boxes):
                    cls = int(box.cls[0])
                    name = model.names[cls].lower()
                    conf = float(box.conf[0])
                    
                    partial_match = any(q in name for q in queries)
                    conf_match = conf >= threshold
                    if partial_match and conf_match:
                        matched_indices.append(i)
            
            filtered_res = results[0][matched_indices]
        else:
            matched_indices = [i for i, box in enumerate(results[0].boxes) if float(box.conf[0]) >= threshold]
            filtered_res = results[0][matched_indices]

        annotated_img = filtered_res.plot()

        objects = []
        for box in filtered_res.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            name = model.names[cls]
            objects.append({"name": name, "confidence": conf})

        _, buffer = cv2.imencode('.jpg', annotated_img)
        image_bytes = buffer.tobytes()

        return image_bytes, img.shape, objects
