import cv2
import numpy as np
import torch
import time
import gc
import threading
from Detection.model_base import shared_model as model
from Services.query_matcher import parse_queries, match_class

# Frame counter for periodic GC
gc_counter = 0
gc_lock = threading.Lock()

def process_webcam_frame_bytes(img_bytes, query=None, threshold=0.25):
    """
    Stateless frame byte processor.
    Decodes frame, runs YOLO prediction, filters by target classes,
    draws premium red overlay boxes, and returns annotated frame + telemetry metrics.
    """
    global gc_counter
    
    # 1. Decode frame in memory
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return None, [], {}
        
    height, width, _ = frame.shape
    resolution_str = f"{width}x{height}"
    
    # 2. PyTorch Inference & Timing
    start_time = time.time()
    with torch.no_grad():
        results = model(frame, verbose=False)
    inference_time = int((time.time() - start_time) * 1000)
    
    # 3. Parse query terms
    targets = parse_queries(query) if query else []
    
    # 4. Filter indices by class match & confidence threshold
    matched_indices = []
    for i, box in enumerate(results[0].boxes):
        cls = int(box.cls[0])
        name = model.names[cls]
        conf = float(box.conf[0])
        
        if conf >= threshold and match_class(name, targets):
            matched_indices.append(i)
            
    filtered_res = results[0][matched_indices]
    
    # 5. Render bounding boxes
    if query:
        # Manual premium Red bounding boxes (BGR is 0, 0, 255)
        annotated_frame = frame.copy()
        for box in filtered_res.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            name = model.names[cls]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"{name} {int(conf * 100)}%"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_y = y1 if y1 - 20 >= 0 else y1 + 20
            cv2.rectangle(annotated_frame, (x1, label_y - 20), (x1 + w + 10, label_y), (0, 0, 255), -1)
            cv2.putText(annotated_frame, label, (x1 + 5, label_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        # Standard plotfallback
        annotated_frame = filtered_res.plot()
        
    # 6. Accumulate telemetry metrics (aggregate duplicate detections)
    counts = {}
    for box in filtered_res.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        name = model.names[cls]
        
        if name not in counts:
            counts[name] = {"name": name, "count": 0, "max_confidence": 0.0}
        counts[name]["count"] += 1
        counts[name]["max_confidence"] = max(counts[name]["max_confidence"], conf)
        
    objects_list = list(counts.values())
    
    # 7. Encode image back to JPEG bytes
    ret, buffer = cv2.imencode('.jpg', annotated_frame)
    if not ret:
        return None, [], {}
        
    result_bytes = buffer.tobytes()
    
    # Timing Metadata
    meta = {
        "inference_time_ms": inference_time,
        "processing_resolution": resolution_str,
        "server_timestamp": int(time.time() * 1000)
    }
    
    # Explicit buffer dereferences to free memory
    del frame
    del annotated_frame
    del results
    del buffer
    
    # 8. Counter-based garbage collection to avoid overhead
    with gc_lock:
        gc_counter += 1
        if gc_counter >= 100:
            gc.collect()
            gc_counter = 0
            
    return result_bytes, objects_list, meta


# ==========================================
# Legacy Mocks for Backward Compatibility
# ==========================================

def start_webcam(query=None, threshold=None):
    """Deprecated: Mocking to prevent startup exceptions"""
    return True

def stop_webcam():
    """Deprecated: Mocking to prevent startup exceptions"""
    pass

def generate_frames():
    """Deprecated: Mocking to prevent startup exceptions"""
    yield b''

def latest_results():
    """Deprecated: Mocking to prevent startup exceptions"""
    return []
