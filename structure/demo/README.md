# Demo Flask service for audio source separation

This demo provides a minimal Flask backend to run either ICA (FastICA) or Conv-TasNet separation on uploaded WAV files.

## Quick start (recommended on Windows)

From the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\structure\demo\start_server.ps1
```

`start_server.ps1` creates `structure/demo/.venv` (once), checks dependencies, and installs `requirements.txt` when needed. This avoids repeated startup failures such as `No module named 'torchaudio'`.

## Manual setup

From `structure/demo`:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

## API

- POST `/separate` (multipart/form-data)
- `file`: WAV file
- `method`: `ica` or `conv_tasnet` (default: `ica`)
- `n_components`: (for ICA)
- `model_path`: absolute path to Conv-TasNet checkpoint (for conv_tasnet)

Response: JSON with `outputs` list containing file paths for separated WAVs.

Example curl (ICA):

```bash
curl -F "file=@mixture.wav" -F "method=ica" http://localhost:5000/separate
```

Example curl (Conv-TasNet):

```bash
curl -F "file=@mixture.wav" -F "method=conv_tasnet" -F "model_path=/abs/path/to/model.pth" http://localhost:5000/separate
```

## Notes

- Conv-TasNet requires a model checkpoint saved in the repository's `src` format (package dictionary with hyperparams and `state_dict`).
- For quick testing of ICA, a mono WAV will be simulated into multiple mixtures.
- This is a demo: for production use run tasks in a background worker and validate uploads.
