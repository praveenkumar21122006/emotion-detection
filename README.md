# Real-Time Emotion Detection System

Evaluates **facial expressions** and **voice modulations** during live video streams. Fusion gives a robust emotion estimate.

- **7 emotions**: `angry, disgust, fear, happy, sad, surprise, neutral`
- **Modalities**: Face (Haar Cascade + DeepFace optional) + Voice (MFCC/pitch/energy via librosa)
- **Streaming**: WebSocket at 8-15 FPS, REST endpoints, webcam / RTSP / video file support
- **Fusion**: Late fusion `0.7*face + 0.3*voice`

## Architecture
```
Browser (getUserMedia) --WS /ws/video--> FastAPI --> FacialEmotionDetector.process_frame()
                       --WS /ws/audio-->          --> VoiceEmotionDetector.classify()
                       --WS /ws/fused-->          --> StreamProcessor.fused_emotion()
Static frontend (static/index.html) draws bounding boxes + confidence bars.

Standalone: python run.py --source 0   # OpenCV loop
            python run.py --mode voice --file sample.wav
```

## Quick Start
```bash
pip install -r requirements.txt
# optional higher accuracy
pip install deepface librosa tensorflow

# 1) Launch server
python run.py --mode server --port 8000
# open http://localhost:8000

# 2) Standalone webcam demo (no server)
python run.py --source 0

# 3) Voice file
python run.py --mode voice --file sample.wav

# 4) RTSP/IP camera
python run.py --source rtsp://user:pass@ip:554/stream
```

## API
- `GET /` → frontend
- `GET /health` → capability check
- `POST /api/detect/face` (multipart image) → `{faces:[{bbox,emotion,confidence,scores}], count, fps}`
- `POST /api/detect/voice` (multipart wav) → `{emotion, confidence, scores, features}`
- `WS /ws/video` → send `{"frame":"data:image/jpeg;base64,..."}` receive face result
- `WS /ws/audio` → send `{"audio":"base64 PCM16","sr":16000}` receive voice result
- `WS /ws/fused` → send `{"frame":..., "audio":..., "sr":16000}` receive fused

## Files
- `app/facial_emotion.py:1` — Haar + DeepFace/heuristic with smoothing
- `app/voice_emotion.py:1` — librosa + numpy fallback prosody classifier
- `app/stream_processor.py:1` — OpenCV capture + async WS helpers + fusion
- `app/main.py:1` — FastAPI + 3 websockets + REST
- `static/index.html:1` — Live demo UI
- `config.py:1` — thresholds
- `run.py:1` — CLI entry

## Notes
- Without `deepface`/`librosa` the system runs in lightweight heuristic mode (no heavy weights).
- For production, replace `predict_heuristic` with a FER model (e.g., `fer`, `deepface` with `emotion` action, or HuggingFace `trpakov/vit-face-expression`).
- Voice model can be swapped for `superb/hubert` or `j-hartmann/emotion-english-distilroberta-base` on transcripts.
