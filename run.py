"""
Standalone runner for webcam / video file without server.
Usage:
  python run.py                # webcam
  python run.py --source 0
  python run.py --source video.mp4 --no-display --save out.mp4
  python run.py --mode voice --file sample.wav
  python run.py --mode server  # launch FastAPI
"""
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Real-Time Emotion Detection")
    parser.add_argument("--source", default=0, help="0 for webcam, or path/RTSP URL")
    parser.add_argument("--no-display", action="store_true", help="headless (no imshow)")
    parser.add_argument("--save", default=None, help="save output video path")
    parser.add_argument("--mode", choices=["video","voice","server"], default="video")
    parser.add_argument("--file", default=None, help="audio file for voice mode")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.mode == "server":
        import uvicorn
        uvicorn.run("app.main:app", host=args.host, port=args.port, reload=True)
        return

    if args.mode == "voice":
        import numpy as np
        from app.voice_emotion import VoiceEmotionDetector
        det = VoiceEmotionDetector()
        if args.file:
            try:
                import soundfile as sf
                y, sr = sf.read(args.file)
                if y.ndim > 1: y = y.mean(axis=1)
                print(f"Loaded {args.file} sr={sr} len={len(y)}")
                # resample to 16k if needed
                if sr != 16000:
                    try:
                        import librosa
                        y = librosa.resample(y.astype(float), orig_sr=sr, target_sr=16000)
                    except Exception:
                        pass
            except Exception as e:
                print(f"Failed to load {args.file}: {e}")
                # fallback: read raw
                import pathlib
                raw = pathlib.Path(args.file).read_bytes()
                y = np.frombuffer(raw, dtype=np.int16).astype(np.float32)/32768.0
        else:
            # synthesize test tone
            sr = 16000
            t = np.linspace(0, 2, 2*sr)
            y = (0.3*np.sin(2*np.pi*200*t)).astype(np.float32)
            print("No --file, using synthetic 200Hz tone")
        res = det.classify(y)
        print(res)
        return

    # video mode
    from app.stream_processor import StreamProcessor
    sp = StreamProcessor()
    # coerce source to int if digit
    src = args.source
    try:
        src = int(src)
    except ValueError:
        pass
    sp.process_video_file(source=src, display=not args.no_display, save_path=args.save)

if __name__ == "__main__":
    main()
