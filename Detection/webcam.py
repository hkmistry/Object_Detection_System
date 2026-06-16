import cv2
import threading
import time
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Global state
camera = None
frame_lock = threading.Lock()
latest_frame = None
latest_objects = []
running = False
search_query = None
confidence_threshold = 0.25


def start_webcam(query=None, threshold=None):
    global camera, running, search_query, confidence_threshold
    search_query = query if query else None
    if threshold is not None:
        confidence_threshold = float(threshold)
    if running:
        return True

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Could not open webcam")
        return False

    running = True
    thread = threading.Thread(target=_capture_loop, daemon=True)
    thread.start()
    return True


def stop_webcam():
    global running, search_query
    running = False
    search_query = None


def _capture_loop():
    global camera, running, latest_frame, latest_objects, search_query, confidence_threshold
    try:
        while running and camera.isOpened():
            ret, frame = camera.read()
            if not ret:
                continue

            results = model(frame, verbose=False)

            # Parse tags/queries
            if search_query:
                queries = [q.strip().lower() for q in search_query.split(',') if q.strip()]
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
                    conf_match = conf >= confidence_threshold
                    if exact_match and conf_match:
                        matched_indices.append(i)
                
                # Phase 2: Substring fallback if no exact matches found
                if not matched_indices:
                    for i, box in enumerate(results[0].boxes):
                        cls = int(box.cls[0])
                        name = model.names[cls].lower()
                        conf = float(box.conf[0])
                        
                        partial_match = any(q in name for q in queries)
                        conf_match = conf >= confidence_threshold
                        if partial_match and conf_match:
                            matched_indices.append(i)
                
                filtered_res = results[0][matched_indices]
            else:
                # If no query filter, still filter by confidence threshold
                matched_indices = [i for i, box in enumerate(results[0].boxes) if float(box.conf[0]) >= confidence_threshold]
                filtered_res = results[0][matched_indices]

            # If search_query is set, draw matching boxes in RED manually; else fallback to standard plot
            if search_query:
                annotated_frame = frame.copy()
                for box in filtered_res.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    name = model.names[cls]
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Red bounding box (BGR is 0, 0, 255)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    
                    # Premium filled text tag background
                    label = f"{name} {int(conf * 100)}%"
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    label_y = y1 if y1 - 20 >= 0 else y1 + 20
                    cv2.rectangle(annotated_frame, (x1, label_y - 20), (x1 + w + 10, label_y), (0, 0, 255), -1)
                    cv2.putText(annotated_frame, label, (x1 + 5, label_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                annotated_frame = filtered_res.plot()

            # Extract detected objects from filtered results
            objects = []
            for box in filtered_res.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                name = model.names[cls]
                objects.append({"name": name, "confidence": conf})

            with frame_lock:
                latest_frame = annotated_frame
                latest_objects = objects
    finally:
        if camera:
            camera.release()
            camera = None
        with frame_lock:
            latest_frame = None
            latest_objects = []



def generate_frames():
    global latest_frame, running
    while running:
        with frame_lock:
            current_frame = latest_frame
            
        if current_frame is None:
            time.sleep(0.05)
            continue
            
        ret, buffer = cv2.imencode('.jpg', current_frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')



def latest_results():
    global latest_objects
    with frame_lock:
        return latest_objects.copy()
