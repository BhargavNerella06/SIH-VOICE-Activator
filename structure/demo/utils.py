import os
import numpy as np
import soundfile as sf
import librosa


def ensure_mono(y):
    # convert multi-channel to mono by averaging channels
    if y.ndim == 1:
        return y
    return np.mean(y, axis=1)


def resample_if_needed(y, orig_sr, target_sr):
    if orig_sr == target_sr:
        return y
    return librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)


def normalize_audio(y, eps=1e-8):
    m = np.max(np.abs(y))
    if m < eps:
        return y
    return y / (m + eps)


def save_wav_int16(path, y, sr):
    # normalize to -1..1 then write int16 PCM
    y = normalize_audio(y)
    y_int16 = (y * 32767).astype('int16')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, y_int16, sr, subtype='PCM_16')
