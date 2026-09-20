import numpy as np
import time
from config import EMOTIONS

class VoiceEmotionDetector:
    """
    Real-time voice modulation analysis.
    Extracts prosodic features (MFCC, pitch, energy, ZCR) and maps to emotions.
    Uses librosa if available, else lightweight numpy fallback.
    """

    def __init__(self, sample_rate=16000):
        self.sr = sample_rate
        self.librosa_available = False
        try:
            import librosa  # type: ignore
            self.librosa = librosa
            self.librosa_available = True
        except Exception:
            self.librosa_available = False

    def _features_librosa(self, y: np.ndarray):
        import librosa
        y = y.astype(np.float32)
        # Trim silence
        y, _ = librosa.effects.trim(y, top_db=30)
        if len(y) < 512:
            return None
        mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=self.sr), axis=1)
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        rms = float(np.mean(librosa.feature.rms(y=y)))
        # Pitch via piptrack
        pitches, mags = librosa.piptrack(y=y, sr=self.sr)
        mask = mags > np.median(mags)
        pitch_vals = pitches[mask]
        pitch_mean = float(np.mean(pitch_vals)) if pitch_vals.size else 0.0
        pitch_std = float(np.std(pitch_vals)) if pitch_vals.size else 0.0
        # Spectral centroid (brightness)
        cent = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=self.sr)))
        return {
            "mfcc_mean": mfcc_mean,
            "mfcc_std": mfcc_std,
            "zcr": zcr,
            "rms": rms,
            "pitch_mean": pitch_mean,
            "pitch_std": pitch_std,
            "centroid": cent,
            "chroma": chroma,
        }

    def _features_numpy(self, y: np.ndarray):
        y = y.astype(np.float32)
        if len(y) == 0:
            return None
        # Energy
        rms = float(np.sqrt(np.mean(y**2)))
        zcr = float(((y[:-1] * y[1:]) < 0).sum() / max(len(y), 1))
        # Zero-padded FFT for coarse spectral centroid
        mag = np.abs(np.fft.rfft(y))
        freqs = np.fft.rfftfreq(len(y), 1 / self.sr)
        centroid = float(np.sum(freqs * mag) / max(np.sum(mag), 1e-9))
        # Pitch proxy: autocorrelation peak
        corr = np.correlate(y, y, mode="full")[len(y) - 1 :]
        # Find peak in 50-400 Hz range
        min_lag = int(self.sr / 400)
        max_lag = int(self.sr / 50)
        if len(corr) > max_lag:
            segment = corr[min_lag:max_lag]
            peak = int(np.argmax(segment)) + min_lag
            pitch_mean = float(self.sr / max(peak, 1))
        else:
            pitch_mean = 0.0
        return {
            "mfcc_mean": np.zeros(13),
            "mfcc_std": np.zeros(13),
            "zcr": zcr,
            "rms": rms,
            "pitch_mean": pitch_mean,
            "pitch_std": 20.0,
            "centroid": centroid,
            "chroma": np.zeros(12),
        }

    def extract_features(self, y: np.ndarray):
        if self.librosa_available:
            feats = self._features_librosa(y)
            if feats is not None:
                return feats
        return self._features_numpy(y)

    def classify(self, y: np.ndarray) -> dict:
        feats = self.extract_features(y)
        if feats is None:
            return {
                "emotion": "neutral",
                "confidence": 0.5,
                "scores": {e: (0.5 if e == "neutral" else 0.08) for e in EMOTIONS},
                "features": {},
                "timestamp": time.time(),
            }

        rms = feats["rms"]
        zcr = feats["zcr"]
        pitch = feats["pitch_mean"]
        pitch_std = feats["pitch_std"]
        centroid = feats["centroid"]

        scores = {e: 0.05 for e in EMOTIONS}

        # Rule mapping validated on prosody literature:
        # happy: high energy, high pitch, high centroid, moderate zcr
        # sad: low energy, low pitch, low centroid
        # angry: high energy, high pitch_std, high zcr
        # fear: high pitch, high pitch_std, moderate energy
        # surprise: very high pitch, high energy, high centroid
        # neutral: mid values

        # Normalize cues
        energy = np.clip(rms * 10, 0, 1)  # rms ~0.02-0.15 typical
        is_loud = rms > 0.06
        is_quiet = rms < 0.025
        is_high_pitch = pitch > 180
        is_low_pitch = 0 < pitch < 130
        is_bright = centroid > 2500
        is_dark = centroid < 1200

        if is_loud and is_high_pitch and is_bright:
            scores["happy"] = 0.45
            scores["surprise"] = 0.25
            scores["angry"] = 0.15
        elif is_loud and pitch_std > 40 and zcr > 0.12:
            scores["angry"] = 0.55
            scores["happy"] = 0.15
        elif is_high_pitch and pitch_std > 35:
            scores["fear"] = 0.45
            scores["surprise"] = 0.30
        elif is_quiet and is_low_pitch and is_dark:
            scores["sad"] = 0.55
            scores["neutral"] = 0.25
        elif is_quiet and not is_high_pitch:
            scores["neutral"] = 0.50
            scores["sad"] = 0.25
        elif pitch > 220 and energy > 0.5:
            scores["surprise"] = 0.50
            scores["happy"] = 0.25
        else:
            scores["neutral"] = 0.45
            # distribute remainder by energy/pitch
            if energy > 0.4:
                scores["happy"] = 0.25

        # Boost based on ZCR (noisy -> angry/fear)
        if zcr > 0.15:
            scores["angry"] += 0.1
            scores["fear"] += 0.05

        total = sum(scores.values())
        for k in scores:
            scores[k] /= total

        dominant = max(scores, key=scores.get)
        return {
            "emotion": dominant,
            "confidence": round(float(scores[dominant]), 3),
            "scores": {k: round(float(v), 3) for k, v in scores.items()},
            "features": {
                "rms": round(float(rms), 4),
                "zcr": round(float(zcr), 4),
                "pitch_mean": round(float(pitch), 1),
                "pitch_std": round(float(pitch_std), 1),
                "centroid": round(float(centroid), 1),
            },
            "timestamp": time.time(),
        }

    def decode_pcm16(self, b64_or_bytes, sample_rate=None) -> np.ndarray:
        """Decode base64 PCM16 mono or raw bytes to float32 normalized array."""
        import base64
        sr = sample_rate or self.sr
        if isinstance(b64_or_bytes, str):
            # handle data URL
            if "," in b64_or_bytes:
                b64_or_bytes = b64_or_bytes.split(",", 1)[1]
            try:
                raw = base64.b64decode(b64_or_bytes)
            except Exception:
                raw = b64_or_bytes.encode()
        else:
            raw = bytes(b64_or_bytes)
        # Try interpret as PCM16
        try:
            arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        except Exception:
            arr = np.zeros(1024, dtype=np.float32)
        # Resample stub: if sr mismatch, simple linear interpolation
        if sr != self.sr and len(arr) > 0:
            duration = len(arr) / sr
            new_len = int(duration * self.sr)
            if new_len > 0:
                x_old = np.linspace(0, 1, len(arr))
                x_new = np.linspace(0, 1, new_len)
                arr = np.interp(x_new, x_old, arr).astype(np.float32)
        return arr
