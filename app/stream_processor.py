import cv2
import asyncio
import time
from collections import deque
import numpy as np
from .facial_emotion import FacialEmotionDetector
from .voice_emotion import VoiceEmotionDetector

class StreamProcessor:
    """Handles live video file / webcam / RTSP stream with both modalities."""

    def __init__(self):
        self.face = FacialEmotionDetector()
        self.voice = VoiceEmotionDetector()
        self.fps_buffer = deque(maxlen=30)
        self.voice_buffer = deque(maxlen=10)

    def process_video_file(self, source=0, display=True, save_path=None):
        """
        source: 0 for webcam, or path to video, or rtsp/http url
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source {source}")

        writer = None
        if save_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = cap.get(cv2.CAP_PROP_FPS) or 20
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))

        print(f"[Stream] Opened source {source}. Press 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            result = self.face.process_frame(frame)
            # Overlay stats
            cv2.putText(frame, f"Faces: {result['count']} FPS:{result['fps']}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            if display:
                cv2.imshow("Real-Time Emotion Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if writer:
                writer.write(frame)

        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()

    async def process_websocket_frame(self, b64_frame: str) -> dict:
        """Async wrapper for WS frame processing."""
        loop = asyncio.get_event_loop()
        frame = await loop.run_in_executor(None, self.face.decode_base64_frame, b64_frame)
        if frame is None:
            return {"error": "invalid frame", "timestamp": time.time()}
        result = await loop.run_in_executor(None, self.face.process_frame, frame)
        return result

    async def process_websocket_audio(self, audio_b64: str, sr=16000) -> dict:
        loop = asyncio.get_event_loop()
        y = await loop.run_in_executor(None, self.voice.decode_pcm16, audio_b64, sr)
        result = await loop.run_in_executor(None, self.voice.classify, y)
        return result

    def fused_emotion(self, face_result: dict, voice_result: dict, alpha=0.7) -> dict:
        """
        Late fusion: alpha*face + (1-alpha)*voice
        """
        if not face_result.get("faces"):
            return voice_result
        # Average face scores across detected faces
        from config import EMOTIONS
        avg_face = {e: 0.0 for e in EMOTIONS}
        for f in face_result["faces"]:
            for e, v in f["scores"].items():
                avg_face[e] += v
        for e in avg_face:
            avg_face[e] /= max(len(face_result["faces"]), 1)

        fused = {}
        for e in EMOTIONS:
            fused[e] = alpha * avg_face[e] + (1 - alpha) * voice_result["scores"][e]
        dom = max(fused, key=fused.get)
        return {
            "emotion": dom,
            "confidence": round(fused[dom], 3),
            "scores": {k: round(v, 3) for k, v in fused.items()},
            "face": face_result,
            "voice": voice_result,
        }
