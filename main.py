import cv2, threading, time, json, numpy as np, httpx, os
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from tensorflow import keras

app = FastAPI()

# --- CORS Middleware ---
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- AI Model Loading ---
try:
    model = keras.models.load_model("frame_cnn_model_one.keras")
    print("Keras model 'frame_cnn_model_one.keras' loaded successfully.")
except Exception as e:
    print(f"FATAL ERROR: Could not load Keras model. AI detection will be disabled. Error: {e}")
    model = None

alerts, drone_sources = {}, {}   # drone_sources will hold both RTMP + HLS

# --- Root route for Render health check ---
@app.get("/")
def root():
    return {"message": "Backend is running", "status": "ok"}

@app.get("/system/status")
def get_system_status():
    status = "online" if model else "offline"
    return {"ai_model_status": status}

def process_stream(drone_id,rtmp_url):
    if not model:
        print(f"[{drone_id}] Skipping processing: ML model is not available.")
        alerts[drone_id] = {"alert": "Error: Model not loaded", "score": 0.0}
        return
    cap = cv2.VideoCapture(rtmp_url)
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"[{drone_id}] Stream disconnected. Will attempt to reconnect in 5 seconds...")
            time.sleep(5)
            cap.release()
            cap = cv2.VideoCapture(video_source)
            continue
        try:
            frame_resized = cv2.resize(frame, (128, 128))
            frame_norm = frame_resized.astype("float32") / 255.0
            pred = model.predict(np.expand_dims(frame_norm, 0), verbose=0)[0][0]
            alerts[drone_id] = {
                "alert": "Violence detected" if pred > 0.5 else "Safe",
                "score": float(pred),
            }
        except Exception as e:
            print(f"[{drone_id}] Error during frame processing: {e}")
            alerts[drone_id] = {"alert": "Processing Error", "score": 0.0}
        time.sleep(1)

def start_all_drones():
    global drone_sources
    drones_file = "drones.json"
    if not os.path.exists(drones_file):
        print(f"ERROR: '{drones_file}' not found. No drones will be started.")
        return
    with open(drones_file) as f:
        drones = json.load(f)
    print(f"Found {len(drones)} drone(s) in '{drones_file}'.")
    for d in drones:
        drone_id = d.get("drone_id")
        rtmp_url = d.get("rtmp_url")
        hls_url = d.get("hls_url")
        if not drone_id or not rtmp_url or not hls_url:
            print(f"Skipping invalid drone entry in config: {d}")
            continue
        # Store both RTMP + HLS for later use
        drone_sources[drone_id] = {"rtmp_url": rtmp_url, "hls_url": hls_url}
        print(f"Starting processing thread for drone '{drone_id}'...")
        threading.Thread(
            target=process_stream,
            args=(drone_id, rtmp_url),   # AI always uses RTMP
            daemon=True,
        ).start()

@app.on_event("startup")
def startup_event():
    print("Application starting up. Initializing drone streams...")
    start_all_drones()

@app.get("/alerts/{drone_id}")
def get_alert(drone_id: str):
    return alerts.get(drone_id, {"alert": "No data", "score": 0.0})

@app.get("/stream/{drone_id}.m3u8")
async def proxy_m3u8(drone_id: str):
    drone = drone_sources.get(drone_id)
    if not drone:
        return Response(content='{"error": "Unknown drone_id"}', status_code=404, media_type="application/json")
    url = drone["hls_url"]   # Always proxy HLS for frontend
    print(f"Proxying m3u8 request for '{drone_id}' to {url}")
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(url)
            r.raise_for_status()
            return Response(content=r.content, media_type="application/vnd.apple.mpegurl")
    except httpx.RequestError as e:
        print(f"Error proxying m3u8 for '{drone_id}': {e}")
        return Response(content=f'{{"error": "Could not fetch stream manifest: {e}"}}',
                        status_code=502, media_type="application/json")

@app.get("/stream/{segment:path}")
async def proxy_segment(segment: str):
    base_url = "http://52.230.105.110:8080/hls/"
    segment_url = base_url + segment
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(segment_url)
            r.raise_for_status()
            return Response(content=r.content, media_type="video/MP2T")
    except httpx.RequestError as e:
        print(f"Error proxying segment '{segment}': {e}")
        return Response(content=f'{{"error": "Could not fetch stream segment: {e}"}}',
                        status_code=502, media_type="application/json")
