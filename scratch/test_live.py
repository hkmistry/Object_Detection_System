import requests

url = "https://object-detection-system-iqj0.onrender.com/upload_image"
image_path = "uploads/Image_Detection.jpg"

print(f"Sending POST request to {url} with {image_path}...")
try:
    with open(image_path, 'rb') as f:
        files = {'image': (image_path, f, 'image/jpeg')}
        data = {'query': '', 'threshold': '0.25', 'raw': 'false'}
        response = requests.post(url, files=files, data=data)
    
    print("Response status code:", response.status_code)
    try:
        res_data = response.json()
        print("Response JSON keys:", list(res_data.keys()))
        print("success:", res_data.get("success"))
        if not res_data.get("success"):
            print("error:", res_data.get("error"))
        else:
            print("Detected objects length:", len(res_data.get("objects", [])))
            print("Detected objects list:", res_data.get("objects", []))
            print("Image base64 prefix:", res_data.get("image", "")[:100])
    except Exception as e:
        print("Failed to parse JSON response. Content:")
        print(response.text[:1000])
except Exception as e:
    print("Error during request:", e)
