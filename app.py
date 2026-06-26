from flask import Flask, render_template, Response, jsonify, request, send_from_directory
from Detection.webcam import start_webcam, stop_webcam, generate_frames, latest_results
from Detection.image import detect_image
from Detection.video import video_bp
from Services.rate_limiter import RateLimiter
from Utils.helpers import clean_base64_image

app = Flask(__name__, static_folder='Static')
# Restrict request body size to 1MB to prevent memory exhaustion attacks
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024

limiter = RateLimiter(limit=5, window=1.0)

@app.route('/')
def index():
    return render_template('index.html')

# Stateless Webcam Process Route
@app.route('/process_webcam_frame', methods=['POST'])
def process_webcam_frame_route():
    # 1. Enforce IP-based rate limits
    ip_addr = request.remote_addr or "0.0.0.0"
    if not limiter.is_allowed(ip_addr):
        return jsonify({"success": False, "error": "Too many frame requests. Rate limit exceeded (Max 5 frames/sec)"}), 429

    # 2. Parse payload JSON
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"success": False, "error": "No image data provided"}), 400

    base64_img = clean_base64_image(data['image'])
    if not base64_img:
        return jsonify({"success": False, "error": "Malformed Base64 payload or unsupported image type"}), 400

    query = data.get('query', '')
    threshold_val = data.get('threshold', 0.25)
    try:
        threshold = float(threshold_val)
    except (ValueError, TypeError):
        threshold = 0.25

    # 3. Decode base64 and process frame
    try:
        import base64
        img_bytes = base64.b64decode(base64_img)
        
        from Detection.webcam import process_webcam_frame_bytes
        annotated_bytes, objects, meta = process_webcam_frame_bytes(img_bytes, query, threshold)
        
        if annotated_bytes is None:
            return jsonify({"success": False, "error": "Failed to decode/process frame"}), 422
            
        encoded_img = base64.b64encode(annotated_bytes).decode('utf-8')
        img_url = f"data:image/jpeg;base64,{encoded_img}"
        
        return jsonify({
            "success": True,
            "data": {
                "image": img_url,
                "objects": objects
            },
            "meta": meta
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

# Legacy web-cam routes (Deprecated/Mocked)
@app.route('/start_webcam')
def start_webcam_route():
    return jsonify({"success": True, "deprecated": True})

@app.route('/stop_webcam')
def stop_webcam_route():
    return jsonify({"success": True, "deprecated": True})

@app.route('/video_feed')
def video_feed():
    return "Webcam stream now processes client-side. Please refresh page.", 410

@app.route('/webcam_data')
def webcam_data():
    return jsonify([])


@app.route('/update_search_threshold')
def update_search_threshold():
    threshold_val = request.args.get('threshold')
    if threshold_val is not None:
        t = float(threshold_val)
        import Detection.webcam as webcam
        webcam.confidence_threshold = t
        from Detection.video import session
        with session.lock:
            session.confidence_threshold = t
        return jsonify({"success": True, "threshold": t})
    return jsonify({"success": False, "error": "Missing threshold parameter"})

@app.route('/update_search_query')
def update_search_query():
    query = request.args.get('query', '')
    import Detection.webcam as webcam
    webcam.search_query = query if query else None
    from Detection.video import session
    with session.lock:
        session.query = query if query else None
    return jsonify({"success": True, "query": query})


#image route
@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "No file part"})
    file = request.files['image']
    if file.filename == '':
        return jsonify({"success": False, "error": "No selected file"})

    query = request.form.get('query', '')
    threshold_val = request.form.get('threshold', None)
    threshold = float(threshold_val) if threshold_val is not None else 0.25
    raw_mode = request.form.get('raw', 'false').lower() == 'true'

    image_bytes, shape, objects = detect_image(file, query=query, threshold=threshold, raw=raw_mode)
    if image_bytes is None:
        return jsonify({"success": False, "error": "Invalid file type"})

    import base64
    encoded_img = base64.b64encode(image_bytes).decode('utf-8')
    img_data = f"data:image/jpeg;base64,{encoded_img}"

    return jsonify({
        "success": True, 
        "image": img_data, 
        "objects": objects,
        "width": shape[1] if shape else None,
        "height": shape[0] if shape else None
    })

#Video Route
app.register_blueprint(video_bp)

@app.route('/static/js/webcam_client.js')
def serve_webcam_client():
    import os
    folder = 'Static/js' if os.path.exists('Static/js') else 'static/js'
    return send_from_directory(folder, 'webcam_client.js')

@app.route('/uploads/<path:filename>')
def serve_uploads(filename):
    return send_from_directory('uploads', filename)

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
