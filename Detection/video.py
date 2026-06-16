from flask import Blueprint, request, jsonify, Response
from ultralytics import YOLO
import cv2, os, base64, threading, time

video_bp = Blueprint('video', __name__)
model = YOLO("yolov8n.pt")

class VideoDetectionSession:
    def __init__(self):
        self.video_path = None
        self.running = False
        self.latest_frame = None
        self.latest_objects = []
        self.video_status = "idle"  # "idle", "ready", "processing", "completed", "error"
        self.session_objects = {}   # Accumulator map for full-session summary
        self.total_frames = 0
        self.current_frame = 0
        self.video_duration = 0
        self.mode = "fast"
        self.query = None
        self.confidence_threshold = 0.25
        self.lock = threading.Lock()
        self.thread = None
        self.last_annotated_frame_b64 = ""

    def reset(self):
        with self.lock:
            self.running = False
            self.video_path = None
            self.latest_frame = None
            self.latest_objects = []
            self.video_status = "idle"
            self.session_objects = {}
            self.total_frames = 0
            self.current_frame = 0
            self.video_duration = 0
            self.query = None
            self.confidence_threshold = 0.25
            self.last_annotated_frame_b64 = ""

    def start_processing(self, video_path, mode="fast", query=None, threshold=None):
        # Stop any currently active processing thread first
        self.stop_processing()
        
        with self.lock:
            self.video_path = video_path
            self.mode = mode
            self.query = query
            if threshold is not None:
                self.confidence_threshold = float(threshold)
            else:
                self.confidence_threshold = 0.25
            self.session_objects = {}
            self.latest_objects = []
            self.latest_frame = None
            self.current_frame = 0
            self.last_annotated_frame_b64 = ""
            
        # Probe total frame count
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            with self.lock:
                self.video_status = "error"
            return False
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total / fps if fps > 0 else 0
        cap.release()
        
        with self.lock:
            self.total_frames = total
            self.video_duration = duration
            self.video_status = "ready"
            self.running = True
            
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()
        return True

    def stop_processing(self):
        with self.lock:
            self.running = False
            self.video_status = "idle"
        if self.thread and self.thread.is_alive():
            # Wait briefly for thread to exit loop
            self.thread.join(timeout=0.5)

    def _filter_results(self, results):
        if self.query:
            queries = [q.strip().lower() for q in self.query.split(',') if q.strip()]
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
                conf_match = conf >= self.confidence_threshold
                if exact_match and conf_match:
                    matched_indices.append(i)
            
            # Phase 2: Substring fallback if no exact matches found
            if not matched_indices:
                for i, box in enumerate(results[0].boxes):
                    cls = int(box.cls[0])
                    name = model.names[cls].lower()
                    conf = float(box.conf[0])
                    
                    partial_match = any(q in name for q in queries)
                    conf_match = conf >= self.confidence_threshold
                    if partial_match and conf_match:
                        matched_indices.append(i)
            
            return results[0][matched_indices]
        else:
            # If no query filter, still filter by confidence threshold
            matched_indices = [i for i, box in enumerate(results[0].boxes) if float(box.conf[0]) >= self.confidence_threshold]
            return results[0][matched_indices]

    def _annotate_frame(self, frame, filtered_res):
        if self.query:
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
            return annotated_frame
        else:
            return filtered_res.plot()

    def _process_loop(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            with self.lock:
                self.video_status = "error"
            return
            
        with self.lock:
            self.video_status = "processing"

        # Configurable analysis frame target for Fast Mode
        TARGET_ANALYSIS_FRAMES = 30
        
        if self.mode == 'full':
            frame_skip = 1
        else:
            if self.total_frames <= 0:
                frame_skip = 10
            else:
                frame_skip = max(1, self.total_frames // TARGET_ANALYSIS_FRAMES)

        frame_count = 0
        last_frame = None

        try:
            while self.running and cap.isOpened():
                if frame_skip > 1:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

                ret, frame = cap.read()
                if not ret:
                    # Analyze and preserve the final frame if the loop skipped past it
                    if last_frame is not None and (frame_count % frame_skip != 0):
                        results = model(last_frame, verbose=False)
                        filtered_res = self._filter_results(results)
                        annotated_frame = self._annotate_frame(last_frame, filtered_res)

                        objects = []
                        frame_counts = {}
                        for box in filtered_res.boxes:
                            cls = int(box.cls[0])
                            name = model.names[cls]
                            frame_counts[name] = frame_counts.get(name, 0) + 1

                        for box in filtered_res.boxes:
                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            name = model.names[cls]
                            objects.append({"name": name, "confidence": conf})
                            with self.lock:
                                if name not in self.session_objects:
                                    self.session_objects[name] = {
                                        "count": 1,
                                        "max_conf": conf,
                                        "peak_sim": frame_counts[name]
                                    }
                                else:
                                    self.session_objects[name]["count"] += 1
                                    self.session_objects[name]["max_conf"] = max(self.session_objects[name]["max_conf"], conf)
                                    self.session_objects[name]["peak_sim"] = max(self.session_objects[name]["peak_sim"], frame_counts[name])

                        _, buffer = cv2.imencode('.jpg', annotated_frame)
                        last_b64 = base64.b64encode(buffer).decode('utf-8')

                        with self.lock:
                            self.latest_frame = annotated_frame
                            self.latest_objects = objects
                            self.last_annotated_frame_b64 = last_b64
                    break

                frame_count += frame_skip
                with self.lock:
                    self.current_frame = min(frame_count, self.total_frames)
                last_frame = frame.copy()

                if frame_count % frame_skip == 0 or frame_skip == 1:
                    results = model(frame, verbose=False)
                    filtered_res = self._filter_results(results)
                    annotated_frame = self._annotate_frame(frame, filtered_res)

                    objects = []
                    frame_counts = {}
                    for box in filtered_res.boxes:
                        cls = int(box.cls[0])
                        name = model.names[cls]
                        frame_counts[name] = frame_counts.get(name, 0) + 1

                    for box in filtered_res.boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        name = model.names[cls]
                        objects.append({"name": name, "confidence": conf})
                        with self.lock:
                            if name not in self.session_objects:
                                self.session_objects[name] = {
                                    "count": 1,
                                    "max_conf": conf,
                                    "peak_sim": frame_counts[name]
                                }
                            else:
                                self.session_objects[name]["count"] += 1
                                self.session_objects[name]["max_conf"] = max(self.session_objects[name]["max_conf"], conf)
                                self.session_objects[name]["peak_sim"] = max(self.session_objects[name]["peak_sim"], frame_counts[name])

                    _, buffer = cv2.imencode('.jpg', annotated_frame)
                    current_b64 = base64.b64encode(buffer).decode('utf-8')

                    with self.lock:
                        self.latest_frame = annotated_frame
                        self.latest_objects = objects
                        self.last_annotated_frame_b64 = current_b64

                # Simulate a real playback speed (e.g., ~20-25 FPS scanner rate)
                time.sleep(0.04)

        except Exception as e:
            with self.lock:
                self.video_status = "error"
            print(f"Error in video capture loop: {e}")
        finally:
            cap.release()
            with self.lock:
                if self.video_status == "processing":
                    if self.running:
                        self.video_status = "completed"
                    else:
                        self.video_status = "idle"
                self.running = False

# Instantiate global session object
session = VideoDetectionSession()

@video_bp.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"success": False, "error": "No video uploaded"})

    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({"success": False, "error": "No file selected"})

    # Stop any active session first to release files
    session.stop_processing()

    # Wait up to 5 seconds for the thread to exit and release the file handle
    start_wait = time.time()
    while session.running and (time.time() - start_wait) < 5.0:
        time.sleep(0.05)

    if session.running:
        return jsonify({"success": False, "error": "Previous video processing did not stop in time. Please try again."})

    # Save to uploads/uploaded_video.mp4 (overwriting previously uploaded)
    os.makedirs('uploads', exist_ok=True)
    input_path = os.path.join('uploads', 'uploaded_video.mp4')
    if os.path.exists(input_path):
        try:
            os.remove(input_path)
        except Exception:
            pass

    video_file.save(input_path)

    # Validate file and get total frame count
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return jsonify({"success": False, "error": "Failed to read video file"})
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    session.reset()
    session.video_path = input_path
    session.total_frames = total_frames
    session.video_status = "ready"

    return jsonify({
        "success": True,
        "total_frames": total_frames,
        "video_url": "/uploads/uploaded_video.mp4"
    })

@video_bp.route('/start_video_detection', methods=['POST'])
def start_video_detection():
    mode = request.form.get('mode', 'fast')
    query = request.form.get('query', '')
    threshold_val = request.form.get('threshold', None)
    threshold = float(threshold_val) if threshold_val else 0.25
    if not session.video_path or not os.path.exists(session.video_path):
        return jsonify({"success": False, "error": "No video uploaded or file missing"})

    success = session.start_processing(session.video_path, mode, query, threshold)
    return jsonify({"success": success})

@video_bp.route('/video_detection_feed')
def video_detection_feed():
    return Response(generate_video_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_video_frames():
    # Keep streaming while session is running/processing
    while session.running or session.video_status == "processing":
        with session.lock:
            current_frame = session.latest_frame
            
        if current_frame is None:
            time.sleep(0.03)
            continue
            
        ret, buffer = cv2.imencode('.jpg', current_frame)
        if not ret:
            time.sleep(0.03)
            continue
            
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.04)

@video_bp.route('/video_detection_data')
def video_detection_data():
    with session.lock:
        status = session.video_status
        current = session.current_frame
        total = session.total_frames
        objects = session.latest_objects
        accumulated = session.session_objects.copy()
        last_b64 = session.last_annotated_frame_b64
        video_duration = session.video_duration

    progress_percent = 0
    if total > 0:
        progress_percent = min(100, int((current / total) * 100))

        # Format the summary detections list
        detections_list = [
            {
                "name": k,
                "count": v["count"],
                "max_conf": v["max_conf"],
                "peak_sim": v.get("peak_sim", v["count"])
            }
            for k, v in sorted(accumulated.items(), key=lambda x: x[1]["count"], reverse=True)
        ]
        
        # Identify top object
        top_name = None
        top_count = 0
        if accumulated:
            top_name = max(accumulated, key=lambda k: accumulated[k]["count"])
            top_count = accumulated[top_name]["count"]

        return jsonify({
            "status": status,
            "current_frame": current,
            "total_frames": total,
            "progress_percent": progress_percent,
            "objects": objects,  # Latest frame's bounding box overlays
            "frame": last_b64,   # Final static frame base64 data
            "video_duration": video_duration,
            "summary": {
                "objects": detections_list,
                "unique_count": len(accumulated),
                "top_object": top_name,
                "top_count": top_count
            }
        })
    else:
        return jsonify({
            "status": status,
            "current_frame": current,
            "total_frames": total,
            "progress_percent": progress_percent,
            "objects": objects,
            "accumulated": accumulated
        })

@video_bp.route('/stop_video_detection', methods=['POST'])
def stop_video_detection_endpoint():
    session.stop_processing()
    return jsonify({"success": True})
