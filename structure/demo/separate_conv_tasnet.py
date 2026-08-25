import os
import uuid
import numpy as np
import soundfile as sf
import torch
from utils import ensure_mono, resample_if_needed, save_wav_int16, normalize_audio


def load_conv_tasnet_model(model_path, device='cpu'):
    # project src directory should be on sys.path by Flask app; load package
    package = torch.load(model_path, map_location=device)
    # import local ConvTasNet implementation
    import sys
    repo_root = os.path.dirname(os.path.dirname(__file__))
    src_path = os.path.join(repo_root, '..', 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from conv_tasnet import ConvTasNet
    model = ConvTasNet.load_model_from_package(package)
    model.to(device)
    model.eval()
    return model


def run_conv_tasnet_file(
    filepath,
    model_path,
    detected_speakers=None,
    model_sr=None,
    device='cpu',
    out_dir=None,
    return_details=False,
):
    if out_dir is None:
        out_dir = os.path.join('demo_app', 'outputs')
    os.makedirs(out_dir, exist_ok=True)

    if model_path is None:
        raise ValueError('model_path must be provided for Conv-TasNet')

    y, sr = sf.read(filepath)
    y = ensure_mono(y)

    # Load model and infer expected sample rate if provided by user
    model = load_conv_tasnet_model(model_path, device=device)

    target_sr = model_sr or sr
    if sr != target_sr:
        y = resample_if_needed(y, sr, target_sr)
        sr = target_sr

    y = y.astype('float32')
    y = normalize_audio(y)

    x = torch.from_numpy(y).unsqueeze(0).to(device)  # [1, T]
    target_speakers = int(detected_speakers) if detected_speakers is not None else int(model.C)
    target_speakers = max(1, target_speakers)

    with torch.no_grad():
        est = model(x, target_speakers=target_speakers)  # [1, C_dynamic, T]

    est_np = est.squeeze(0).cpu().numpy()  # (C, T)

    out_paths = []
    for i, s in enumerate(est_np):
        s = normalize_audio(s)
        out_path = os.path.join(out_dir, f'source_conv_{uuid.uuid4().hex[:8]}_{i+1}.wav')
        save_wav_int16(out_path, s, sr)
        out_paths.append(out_path)

    details = {
        "requested_speakers": target_speakers,
        "model_trained_speakers": int(model.C),
        "effective_output_speakers": int(est_np.shape[0]),
    }
    if return_details:
        return out_paths, details
    return out_paths


def run_torchaudio_conv_tasnet_file(
    filepath,
    out_dir=None,
    device='cpu',
    return_details=False,
):
    if out_dir is None:
        out_dir = os.path.join('demo_app', 'outputs')
    os.makedirs(out_dir, exist_ok=True)

    from torchaudio.pipelines import CONVTASNET_BASE_LIBRI2MIX

    bundle = CONVTASNET_BASE_LIBRI2MIX
    model = bundle.get_model().to(device)
    model.eval()

    y, sr = sf.read(filepath)
    y = ensure_mono(y).astype('float32')
    if sr != bundle.sample_rate:
        y = resample_if_needed(y, sr, bundle.sample_rate)
        sr = bundle.sample_rate
    y = normalize_audio(y)

    x = torch.from_numpy(y).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        est = model(x)

    est_np = est.squeeze(0).cpu().numpy()
    if est_np.ndim == 3:
        est_np = est_np[:, 0, :]

    out_paths = []
    for i, s in enumerate(est_np):
        s = normalize_audio(s)
        out_path = os.path.join(out_dir, f'source_torchaudio_conv_{uuid.uuid4().hex[:8]}_{i+1}.wav')
        save_wav_int16(out_path, s, sr)
        out_paths.append(out_path)

    details = {
        "model": "torchaudio.pipelines.CONVTASNET_BASE_LIBRI2MIX",
        "model_sample_rate": int(bundle.sample_rate),
        "effective_output_speakers": int(est_np.shape[0]),
        "note": "Pretrained for 2-speaker speech mixtures; best results need overlapping speech similar to its training data.",
    }
    if return_details:
        return out_paths, details
    return out_paths
