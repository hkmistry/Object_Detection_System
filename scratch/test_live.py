import requests
import base64

# A valid 1x1 blank pixel base64 JPEG
mock_base64_jpeg = (
    "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP///////////////////////"
    "///////////////////////////////////////////////////////////////wgALCAABAAEBAREA/"
    "8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPxA="
)

url = "http://10.240.119.97:5000/process_webcam_frame"
payload = {
    "image": mock_base64_jpeg,
    "query": "person,bottle",
    "threshold": 0.25
}

print(f"Sending POST request to local server: {url}...")
try:
    response = requests.post(url, json=payload, timeout=5)
    print("Response status code:", response.status_code)
    
    res_data = response.json()
    print("Response JSON success:", res_data.get("success"))
    
    if res_data.get("success"):
        print("Telemetry Metadata:")
        print("  Inference Time:", res_data.get("meta", {}).get("inference_time_ms"), "ms")
        print("  Processing Resolution:", res_data.get("meta", {}).get("processing_resolution"))
        print("  Server Timestamp:", res_data.get("meta", {}).get("server_timestamp"))
        
        data_block = res_data.get("data", {})
        print("Detected Objects Count:", len(data_block.get("objects", [])))
        print("Detected Objects:", data_block.get("objects", []))
        print("Annotated Image (Base64 URL prefix):", data_block.get("image", "")[:80] + "...")
        print("\nSUCCESS: Stateless webcam processing endpoint verified locally!")
    else:
        print("ERROR:", res_data.get("error"))
except Exception as e:
    print("FAILED during execution:", e)
