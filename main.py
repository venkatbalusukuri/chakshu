import cv2, threading, time, json, numpy as np, httpx, os
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from tensorflow import keras

app = FastAPI()

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load AI Model ---
try:
    model = keras.models.load_model("frame_cnn_model_one.keras")
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Could not load model: {e}")
    model = None

alerts, drone_sources = {}, {}

# --- Root + Health ---
@app.get("/")
def root():
    return {"message": "Backend running", "status": "ok"}

@app.get("/system/status")
def get_status():
    return {"ai_model_status": "online" if model else "offline"}

# --- AI Processing Thread ---
def process_stream(drone_id, video_source):
    if not model:
        alerts[drone_id] = {"alert": "Error: Model not loaded", "score": 0.0}
        return

    cap = cv2.VideoCapture(video_source)
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"[{drone_id}] ⚠️ No frames. Reconnecting in 5s...")
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
            print(f"[{drone_id}] ❌ Error: {e}")
            alerts[drone_id] = {"alert": "Processing Error", "score": 0.0}
        time.sleep(1)

# --- Startup: Load drones.json ---
def start_all_drones():
    global drone_sources
    if not os.path.exists("drones.json"):
        print("❌ drones.json not found.")
        return
    with open("drones.json") as f:
        drones = json.load(f)
    print(f"📡 Found {len(drones)} drone(s).")
    for d in drones:
        drone_id, video_source = d.get("drone_id"), d.get("video_source")
        if not drone_id or not video_source:
            print(f"⚠️ Skipping invalid entry: {d}")
            continue
        drone_sources[drone_id] = video_source
        print(f"🚀 Starting AI thread for '{drone_id}'")
        threading.Thread(target=process_stream, args=(drone_id, video_source), daemon=True).start()

@app.on_event("startup")
def startup_event():
    print("🔄 Starting drone streams...")
    start_all_drones()

# --- Alerts API ---
@app.get("/alerts/{drone_id}")
def get_alert(drone_id: str):
    return alerts.get(drone_id, {"alert": "No data", "score": 0.0})

@app.get("/alerts/all")
def get_all_alerts():
    return alerts

# --- Proxy API ---
@app.get("/stream/{drone_id}.m3u8")
async def proxy_m3u8(drone_id: str):
    video_source = drone_sources.get(drone_id)
    if not video_source:
        return Response(content='{"error": "Unknown drone_id"}',
                        status_code=404, media_type="application/json")
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(video_source)
            r.raise_for_status()
            return Response(content=r.content, media_type="application/vnd.apple.mpegurl")
    except Exception as e:
        return Response(content=f'{{"error": "Could not fetch manifest: {e}"}}',
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
    except Exception as e:
        return Response(content=f'{{"error": "Could not fetch segment: {e}"}}',
                        status_code=502, media_type="application/json")
