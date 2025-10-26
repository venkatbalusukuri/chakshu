import cv2
import requests
import time

# Backend API endpoint
API = "http://127.0.0.1:8000/predict"   # use http://<VM_PUBLIC_IP>:8000 if running from outside
DRONE_ID = "drone1"

# Path to your video file (or 0 for webcam, or RTSP URL)
VIDEO_PATH = "NV_13.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("❌ Could not open video source:", VIDEO_PATH)
    exit()

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("⚠️ Skipping empty frame")
        break

    frame_count += 1
    print(f"[Client] Processing frame {frame_count}")

    # Encode frame as JPEG
    success, jpg = cv2.imencode(".jpg", frame)
    if not success:
        print("❌ Failed to encode frame, skipping")
        continue

    jpg_bytes = jpg.tobytes()
    print(f"[Client] Encoded frame size: {len(jpg_bytes)} bytes")

    files = {"file": ("frame.jpg", jpg_bytes, "image/jpeg")}
    params = {"drone_id": DRONE_ID}

    try:
        r = requests.post(API, files=files, params=params, timeout=10)
        print("[Server Response]", r.status_code, r.json())
    except Exception as e:
        print("❌ Error sending frame:", e)

    # Send ~2 frames per second
    time.sleep(0.5)

cap.release()
print("✅ Video stream finished")
