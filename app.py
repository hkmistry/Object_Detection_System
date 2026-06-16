from flask import Flask, render_template, Response, jsonify, request, send_from_directory
from Detection.webcam import start_webcam, stop_webcam, generate_frames, latest_results
from Detection.image import detect_image
from Detection.video import video_bp

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

#web-cam route
@app.route('/start_webcam')
def start_webcam_route():
    query = request.args.get('query', '')
    threshold_val = request.args.get('threshold', None)
    threshold = float(threshold_val) if threshold_val is not None else None
    success = start_webcam(query=query, threshold=threshold)
    return jsonify({"success": success})


@app.route('/stop_webcam')
def stop_webcam_route():
    stop_webcam()
    return jsonify({"success": True})


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/webcam_data')
def webcam_data():
    objects = latest_results()
    return jsonify(objects)

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

@app.route('/uploads/<path:filename>')
def serve_uploads(filename):
    return send_from_directory('uploads', filename)

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
