Project folder structure for the Conv-TasNet demo and helpers.

Root: `Conv-TasNet/`

- `structure/demo/`
- `app.py` - Flask app with `/separate` and `/download` endpoints.
- `utils.py` - Audio preprocessing, resampling, and saving helpers.
- `separate_ica.py` - ICA-based separation runner (`sklearn.FastICA`).
- `separate_conv_tasnet.py` - Conv-TasNet runner for checkpoint and torchaudio bundle inference.
- `requirements.txt` - Python dependencies for the demo server.
- `start_server.ps1` - Windows startup helper that bootstraps `.venv` and installs dependencies automatically.
- `README.md` - Setup and API usage for the demo service.
- `uploads/` - Runtime upload directory.
- `outputs/` - Runtime output directory.

- `src/`
- `conv_tasnet.py` - Model implementation.
- `utils.py` - Overlap-add and model helper functions.
- `separate.py` - Repository script for offline separation using trained models.
- `train.py` - Training script.

Notes:
- To run the demo Flask service on Windows from the repo root, use `.\structure\demo\start_server.ps1`.
- The Conv-TasNet runner expects a checkpoint compatible with `src/conv_tasnet.ConvTasNet` (package dictionary with hyperparameters and `state_dict`).
- The ICA runner can simulate mixtures from mono input for quick testing.
