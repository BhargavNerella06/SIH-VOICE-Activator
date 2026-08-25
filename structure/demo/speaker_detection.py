import logging
from typing import Dict, Tuple

import librosa
import numpy as np
import soundfile as sf
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from utils import ensure_mono, normalize_audio, resample_if_needed


LOGGER = logging.getLogger(__name__)


def _segment_audio(
    audio: np.ndarray,
    sr: int,
    segment_seconds: float = 1.5,
    hop_seconds: float = 0.75,
) -> np.ndarray:
    """Slice audio into overlapping segments with shape (num_segments, num_samples)."""
    seg_len = max(1, int(segment_seconds * sr))
    hop_len = max(1, int(hop_seconds * sr))

    if len(audio) < seg_len:
        pad = np.zeros(seg_len - len(audio), dtype=audio.dtype)
        return np.expand_dims(np.concatenate([audio, pad]), axis=0)

    segments = []
    for start in range(0, len(audio) - seg_len + 1, hop_len):
        end = start + seg_len
        segments.append(audio[start:end])
    return np.asarray(segments)


def extract_embeddings(
    audio: np.ndarray,
    sr: int = 16000,
    segment_seconds: float = 1.5,
    hop_seconds: float = 0.75,
    min_rms: float = 0.01,
) -> np.ndarray:
    """
    Extract lightweight speaker-like embeddings from voiced segments.

    Embedding = [MFCC mean/std, delta-MFCC mean/std] for each segment.
    """
    segments = _segment_audio(audio, sr, segment_seconds=segment_seconds, hop_seconds=hop_seconds)
    embeddings = []

    for seg in segments:
        rms = float(np.sqrt(np.mean(np.square(seg)) + 1e-8))
        if rms < min_rms:
            continue

        mfcc = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=13)
        delta = librosa.feature.delta(mfcc)
        emb = np.concatenate(
            [
                mfcc.mean(axis=1),
                mfcc.std(axis=1),
                delta.mean(axis=1),
                delta.std(axis=1),
            ]
        )
        embeddings.append(emb)

    if not embeddings:
        return np.empty((0, 52), dtype=np.float32)
    return np.asarray(embeddings, dtype=np.float32)


def cluster_embeddings(
    embeddings: np.ndarray,
    min_speakers: int = 1,
    max_speakers: int = 6,
    random_state: int = 0,
) -> Tuple[int, Dict[str, float]]:
    """Estimate speaker count using KMeans + silhouette score."""
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("No embeddings available for clustering.")

    n_samples = embeddings.shape[0]
    if n_samples < 3:
        return max(min_speakers, 1), {"reason": "too_few_segments"}

    scaler = StandardScaler()
    X = scaler.fit_transform(embeddings)

    k_min = max(2, min_speakers)
    k_max = min(max_speakers, n_samples - 1)
    if k_min > k_max:
        return max(min_speakers, 1), {"reason": "insufficient_samples_for_silhouette"}

    best_k = k_min
    best_score = -1.0
    scores: Dict[str, float] = {}

    for k in range(k_min, k_max + 1):
        model = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = model.fit_predict(X)
        if len(np.unique(labels)) < 2:
            continue
        score = float(silhouette_score(X, labels))
        scores[f"k={k}"] = score
        if score > best_score:
            best_score = score
            best_k = k

    scores["best_score"] = best_score
    return best_k, scores


def detect_num_speakers(
    audio_path: str,
    target_sr: int = 16000,
    min_speakers: int = 1,
    max_speakers: int = 6,
    fallback_speakers: int = 2,
    debug: bool = True,
) -> Tuple[int, Dict[str, object]]:
    """
    End-to-end speaker count detection.

    Returns:
        (n_speakers, debug_info)
    """
    debug_info: Dict[str, object] = {"audio_path": audio_path}
    try:
        y, sr = sf.read(audio_path)
        y = ensure_mono(y)
        y = y.astype(np.float32)
        if sr != target_sr:
            y = resample_if_needed(y, sr, target_sr)
            sr = target_sr
        y = normalize_audio(y)

        embeddings = extract_embeddings(y, sr=sr)
        debug_info["num_segments"] = int(embeddings.shape[0])

        if embeddings.shape[0] == 0:
            raise ValueError("No voiced segments detected for speaker estimation.")

        n_speakers, scores = cluster_embeddings(
            embeddings,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        n_speakers = int(max(min_speakers, min(max_speakers, n_speakers)))

        debug_info["scores"] = scores
        debug_info["n_speakers"] = n_speakers
        if debug:
            LOGGER.info("Speaker detection succeeded: %s", debug_info)
        return n_speakers, debug_info
    except Exception as exc:
        debug_info["error"] = str(exc)
        debug_info["fallback_n_speakers"] = fallback_speakers
        if debug:
            LOGGER.warning("Speaker detection failed, using fallback: %s", debug_info)
        return int(fallback_speakers), debug_info
