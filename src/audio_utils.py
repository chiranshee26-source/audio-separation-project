import glob
import os

import librosa
import numpy as np
import soundfile as sf

SR = 16000          # sample rate (Hz): 16k is enough for speech and keeps files small
DURATION = 10.0     # seconds: all sources must be the same length


def find_source(name, folder="data/sources"):
    """Find data/sources/<name>.<any extension> so wav/mp3/flac all work."""
    matches = glob.glob(os.path.join(folder, name + ".*"))
    if not matches:
        raise FileNotFoundError(f"No file named {name}.* in {folder}")
    return matches[0]


def load_source(path, sr=SR, duration=DURATION):
    """Load audio as mono, resampled to `sr`, trimmed/padded to `duration`."""
    y, _ = librosa.load(path, sr=sr, mono=True, duration=duration)
    target_len = int(sr * duration)
    if len(y) < target_len:
        print(f"WARNING: {path} is only {len(y)/sr:.1f}s, padding with silence")
        y = np.pad(y, (0, target_len - len(y)))
    y = y / (np.max(np.abs(y)) + 1e-12)          # peak-normalize to [-1, 1]
    return y


def save_wav(path, signal, sr=SR):
    """Save a 1-D signal as WAV, scaled so it never clips."""
    signal = signal / (np.max(np.abs(signal)) + 1e-12) * 0.9
    sf.write(path, signal, sr)