import os
import uuid
import numpy as np
import soundfile as sf
from sklearn.decomposition import FastICA
from utils import ensure_mono, resample_if_needed, save_wav_int16, normalize_audio


def simulate_mixtures_from_mono(y, n_channels):
    # create simple simulated mixtures by scaling the mono signal
    scales = np.linspace(0.6, 1.0, n_channels)
    X = np.stack([y * s for s in scales], axis=0)
    return X


def run_ica_file(filepath, n_components=2, target_sr=16000, out_dir=None):
    if out_dir is None:
        out_dir = os.path.join('demo_app', 'outputs')
    os.makedirs(out_dir, exist_ok=True)

    y, sr = sf.read(filepath)
    # ensure 1-D mono vector for simulation or take channels
    if y.ndim == 2 and y.shape[1] > 1:
        # soundfile returns (T, C); convert to (C, T)
        y_t = y.T
        X = y_t  # shape (C, T)
    else:
        mono = ensure_mono(y)
        if n_components <= 1:
            raise ValueError('ICA requires at least 2 components')
        X = simulate_mixtures_from_mono(mono, n_components)

    # transpose to shape (T, n_channels) for FastICA
    X_t = X.T
    n_features = X_t.shape[1]
    if n_components > n_features:
        n_components = n_features

    # resample each channel if needed (we assume same sr)
    if sr != target_sr:
        X_t = np.stack([resample_if_needed(X_t[:, i], sr, target_sr) for i in range(X_t.shape[1])], axis=1)
        sr = target_sr

    # center/whiten handled by FastICA
    ica = FastICA(n_components=n_components, random_state=0)
    S = ica.fit_transform(X_t)  # shape (T, n_components)
    S = S.T  # (n_components, T)

    out_paths = []
    for i, s in enumerate(S):
        s = normalize_audio(s)
        out_path = os.path.join(out_dir, f'source_ica_{uuid.uuid4().hex[:8]}_{i+1}.wav')
        save_wav_int16(out_path, s, sr)
        out_paths.append(out_path)

    return out_paths
