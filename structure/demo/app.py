import os
import uuid
import logging
import subprocess
import sys
from flask import Flask, request, jsonify, send_file, send_from_directory

from separate_ica import run_ica_file
from speaker_detection import detect_num_speakers

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
OUT_FOLDER = os.path.join(os.path.dirname(__file__), 'outputs')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def ensure_torchaudio_installed():
    try:
        import torchaudio  # noqa: F401
        return False
    except ModuleNotFoundError:
        LOGGER.warning("torchaudio is missing; installing into %s", sys.executable)
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'torchaudio'])
        return True


@app.route('/health')
def health():
    return 'ok'


@app.route('/')
def index():
    return send_from_directory(os.path.join(os.path.dirname(__file__), 'static'), 'upload.html')


@app.route('/separate', methods=['POST'])
def separate():
    # form fields: file (multipart), method (ica|conv_tasnet), model_path
    if 'file' not in request.files:
        return jsonify({'error': 'no file provided'}), 400
    f = request.files['file']
    method = request.form.get('method', 'ica')
    model_path = request.form.get('model_path')

    filename = f.filename or f'mixture_{uuid.uuid4().hex[:8]}.wav'
    saved_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    f.save(saved_path)

    # Automatically detect number of speakers before separation.
    n_speakers, detect_debug = detect_num_speakers(saved_path, fallback_speakers=2, debug=True)
    LOGGER.info("Detected speakers=%s for file=%s", n_speakers, saved_path)

    inference_debug = {}
    try:
        if method == 'ica':
            out_paths = run_ica_file(saved_path, n_components=n_speakers, out_dir=OUT_FOLDER)
            inference_debug = {"effective_output_speakers": len(out_paths)}
        elif method == 'conv_tasnet':
            from separate_conv_tasnet import run_conv_tasnet_file, run_torchaudio_conv_tasnet_file
            if model_path:
                out_paths, inference_debug = run_conv_tasnet_file(
                    saved_path,
                    model_path=model_path,
                    detected_speakers=n_speakers,
                    out_dir=OUT_FOLDER,
                    return_details=True,
                )
            else:
                try:
                    out_paths, inference_debug = run_torchaudio_conv_tasnet_file(
                        saved_path,
                        out_dir=OUT_FOLDER,
                        return_details=True,
                    )
                except ModuleNotFoundError as e:
                    if e.name != 'torchaudio':
                        raise
                    ensure_torchaudio_installed()
                    out_paths, inference_debug = run_torchaudio_conv_tasnet_file(
                        saved_path,
                        out_dir=OUT_FOLDER,
                        return_details=True,
                    )
                    inference_debug = dict(inference_debug)
                    inference_debug['auto_installed_torchaudio'] = True
        else:
            return jsonify({'error': 'unknown method'}), 400
    except ModuleNotFoundError as e:
        if e.name == 'torchaudio':
            return jsonify({
                'error': "torchaudio is missing in the active Python environment. "
                         "Run structure/demo/start_server.ps1 to auto-install demo dependencies."
            }), 500
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # return download URLs (simple paths)
    downloads = [{'name': os.path.basename(p), 'path': p} for p in out_paths]
    return jsonify({
        'detected_speakers': n_speakers,
        'detection_debug': detect_debug,
        'inference_debug': inference_debug,
        'outputs': downloads,
    })


@app.route('/download')
def download():
    # query param path=absolute/path/to/file
    path = request.args.get('path')
    if not path or not os.path.exists(path):
        return 'file not found', 404
    return send_file(path, as_attachment=True)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', '5000'))
    debug = os.environ.get('FLASK_DEBUG', '1') == '1'
    app.run(host='0.0.0.0', port=port, debug=debug)
