"""
Flask Web App — Lecture Summarizer
Pre-loads models at startup. debug=False to prevent reloading loop.
"""
import os, json, logging, tempfile, threading
from pathlib import Path
from flask import Flask, render_template, request, jsonify

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

_pipeline = None
_lock     = threading.Lock()
_ready    = False

ALLOWED_EXT = {"wav","mp3","flac","ogg","m4a","mp4","mkv","avi","mov","webm"}

def get_pipeline():
    global _pipeline, _ready
    if _pipeline is None:
        with _lock:
            if _pipeline is None:
                from pipeline import LectureSummarizationPipeline
                _pipeline = LectureSummarizationPipeline()
                _ready = True
    return _pipeline

def preload_pipeline():
    def _load():
        import time; time.sleep(1)
        logger.info("Pre-loading models in background…")
        get_pipeline()
        logger.info("Models ready!")
    threading.Thread(target=_load, daemon=True).start()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/summarize_url", methods=["POST"])
def summarize_url():
    data = request.get_json(force=True)
    url  = (data.get("url") or "").strip()
    fmt  = data.get("format", "both")
    if not url:
        return jsonify({"error": "No URL provided"}), 400
    try:
        results = get_pipeline().process(url, output_format=fmt)
        return jsonify({"success": True, **results})
    except Exception as e:
        logger.exception("URL pipeline error")
        return jsonify({"error": str(e)}), 500

@app.route("/api/summarize_file", methods=["POST"])
def summarize_file():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    fmt  = request.form.get("format", "both")
    ext  = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXT:
        return jsonify({"error": f"Unsupported file type .{ext}"}), 400
    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name
    try:
        results = get_pipeline().process(tmp_path, output_format=fmt)
        return jsonify({"success": True, **results})
    except Exception as e:
        logger.exception("File pipeline error")
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "models_ready": _ready})

if __name__ == "__main__":
    preload_pipeline()
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)
