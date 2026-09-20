from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64
import json
import time
from pathlib import Path

from .facial_emotion import FacialEmotionDetector
from .voice_emotion import VoiceEmotionDetector
from .stream_processor import StreamProcessor

app = FastAPI(title="Real-Time Emotion Detection", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = FacialEmotionDetector()
voice_detector = VoiceEmotionDetector()
stream_processor = StreamProcessor()

# Mount static
static_dir = Path(__file__).resolve().parent.parent / "static"
static_dir.mkdir(exist_ok=True)
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/", response_class=HTMLResponse)
def index():
    html_path = static_dir / "index.html"
    if html_path.exists():
        return html_path.read_text()
    return HTMLResponse("<h1>Real-Time Emotion Detection API running</h1><p>See /docs</p>")

@app.get("/health")
def health():
    return {"status": "ok", "deepface": detector.deepface_available, "librosa": voice_detector.librosa_available}

@app.post("/api/detect/face")
async def detect_face(file: UploadFile = File(...)):
    data = await file.read()
    arr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return JSONResponse({"error": "invalid image"}, status_code=400)
    result = detector.process_frame(frame)
    return result

@app.post("/api/detect/voice")
async def detect_voice(file: UploadFile = File(...), sr: int = 16000):
    data = await file.read()
    # Assume wav or raw pcm16
    try:
        import soundfile as sf
        import io
        y, file_sr = sf.read(io.BytesIO(data))
        if y.ndim > 1:
            y = y.mean(axis=1)
        y = y.astype(np.float32)
        # resample if needed via librosa or naive
        if file_sr != 16000:
            try:
                import librosa
                y = librosa.resample(y, orig_sr=file_sr, target_sr=16000)
            except Exception:
                pass
    except Exception:
        # fallback: raw PCM16
        y = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    result = voice_detector.classify(y)
    return result

@app.websocket("/ws/video")
async def ws_video(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            msg = await ws.receive_text()
            try:
                payload = json.loads(msg)
                b64 = payload.get("frame") or payload.get("data") or msg
            except json.JSONDecodeError:
                b64 = msg
            # handle bare base64 string without JSON
            frame = detector.decode_base64_frame(b64)
            if frame is None:
                await ws.send_json({"error": "invalid frame"})
                continue
            result = detector.process_frame(frame)
            await ws.send_json(result)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"error": str(e)})
            await ws.close()
        except Exception:
            pass

@app.websocket("/ws/audio")
async def ws_audio(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            msg = await ws.receive_text()
            try:
                payload = json.loads(msg)
                b64 = payload.get("audio") or payload.get("data") or msg
                sr = int(payload.get("sr", 16000))
            except json.JSONDecodeError:
                b64 = msg
                sr = 16000
            y = voice_detector.decode_pcm16(b64, sr)
            result = voice_detector.classify(y)
            await ws.send_json(result)
    except WebSocketDisconnect:
        pass

@app.websocket("/ws/fused")
async def ws_fused(ws: WebSocket):
    """Expects JSON {frame: base64, audio: base64, sr: 16000} -> fused emotion"""
    await ws.accept()
    try:
        while True:
            msg = await ws.receive_text()
            payload = json.loads(msg)
            b64_frame = payload.get("frame")
            b64_audio = payload.get("audio")
            sr = int(payload.get("sr", 16000))
            face_res = {"faces": [], "count": 0}
            voice_res = {"emotion": "neutral", "confidence": 0.5, "scores": {e: 0.14 for e in ["angry","disgust","fear","happy","sad","surprise","neutral"]}}
            if b64_frame:
                frame = detector.decode_base64_frame(b64_frame)
                if frame is not None:
                    face_res = detector.process_frame(frame)
            if b64_audio:
                y = voice_detector.decode_pcm16(b64_audio, sr)
                voice_res = voice_detector.classify(y)
            fused = stream_processor.fused_emotion(face_res, voice_res)
            await ws.send_json(fused)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"error": str(e)})
        except Exception:
            pass
