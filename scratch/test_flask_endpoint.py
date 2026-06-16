import requests

url = "http://127.0.0.1:5000/upload_video"
video_path = "uploads/obj_video_1.mp4"

print(f"Sending POST request to {url} with {video_path}...")
try:
    with open(video_path, 'rb') as f:
        files = {'video': (video_path, f, 'video/mp4')}
        response = requests.post(url, files=files)
    
    print("Response status code:", response.status_code)
    try:
        data = response.json()
        print("Response JSON keys:", list(data.keys()))
        print("success:", data.get("success"))
        if not data.get("success"):
            print("error:", data.get("error"))
        else:
            print("Detected objects length:", len(data.get("objects", [])))
            print("Sample objects:", data.get("objects", [])[:5])
            print("Frame base64 prefix:", data.get("frame", "")[:100])
    except Exception as e:
        print("Failed to parse JSON response. Content:")
        print(response.text[:500])
except Exception as e:
    print("Error during request:", e)
