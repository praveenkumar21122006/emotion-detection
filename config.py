EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

FACIAL_CONFIG = {
    "scaleFactor": 1.1,
    "minNeighbors": 5,
    "minSize": (30, 30),
    "model": "haar",  # haar | dnn
    "confidence_threshold": 0.5,
}

VOICE_CONFIG = {
    "sample_rate": 16000,
    "n_mfcc": 13,
    "frame_duration_ms": 1000,  # analysis window
    "hop_length": 512,
}

STREAM_CONFIG = {
    "fps": 15,
    "jpeg_quality": 80,
    "max_faces": 5,
    "ws_max_frame_size": 2 * 1024 * 1024,
}
