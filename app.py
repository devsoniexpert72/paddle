#!/usr/bin/env python3
"""
OCR host - CPU-only, small-memory friendly.

Usage:
  python ocr_cpu_only.py
Notes:
 - Make sure you installed PaddlePaddle CPU wheel as per Paddle docs and paddleocr.
 - This script forces CPU by clearing CUDA_VISIBLE_DEVICES and setting paddle device to 'cpu'.
"""
import os, sys, time, json, mimetypes, threading
from flask import Flask, send_file, render_template_string, jsonify
from PIL import Image

# ---- Force single-threaded BLAS/OpenMP before heavy imports ----
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

# ---- Force Paddle to use CPU only ----
# This prevents any accidental GPU usage even if a GPU-enabled wheel is present.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# ---- Config ----
HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", 8000))
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")
MAX_DIM = int(os.environ.get("OCR_MAX_DIM", 1024))   # reduce to 800 or 600 if memory spikes
CACHE_FN = "ocr_cache_cpu.json"
USE_ANGLE = False    # disable angle classifier to save memory/time
USE_GPU_FLAG = False # we pass this to PaddleOCR explicitly

HTML_TMPL = """<!doctype html><meta charset="utf-8"><title>OCR - CPU</title>
<style>body{font-family:system-ui;padding:18px}img{max-width:600px;height:auto;border:1px solid #ddd}</style>
<h1>OCR (CPU)</h1><p>Image: <b>{{img_name}}</b></p>
<div style="display:flex;gap:20px;align-items:flex-start">
  <div><img src="/image" alt="image"><p><a href="/image" download>Download</a></p></div>
  <div><h3>Extracted text</h3><pre style="white-space:pre-wrap;background:#f7f7f7;padding:10px;border-radius:6px">{{text}}</pre>
  <p><a href="/api/ocr">JSON output</a></p></div>
</div>
"""

app = Flask(__name__)

# ---- Find first image in current directory ----
def find_image():
    for f in sorted(os.listdir(os.getcwd())):
        if f.lower().endswith(IMAGE_EXTS) and os.path.isfile(f):
            return os.path.abspath(f)
    return None

IMAGE_PATH = find_image()
if not IMAGE_PATH:
    sys.exit("No image found in current directory. Put an image (jpg/png/...) and re-run.")

# ---- Simple disk cache to avoid re-running OCR on restarts ----
cache_lock = threading.Lock()
def load_cache():
    try:
        with open(CACHE_FN, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}
def save_cache(d):
    try:
        with open(CACHE_FN, "w", encoding="utf-8") as fh:
            json.dump(d, fh, ensure_ascii=False)
    except Exception:
        pass

# ---- Downscale big images (reduce memory) ----
def open_and_downscale(path, max_dim=MAX_DIM):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    max_side = max(w, h)
    if max_side > max_dim:
        scale = max_dim / float(max_side)
        new_size = (int(w * scale), int(h * scale))
        img = img.resize(new_size, Image.LANCZOS)
    return img

# ---- Lazy PaddleOCR init (CPU-only) ----
ocr_instance = None
ocr_init_lock = threading.Lock()

def get_ocr():
    global ocr_instance
    if ocr_instance is not None:
        return ocr_instance
    with ocr_init_lock:
        if ocr_instance is not None:
            return ocr_instance
        # Import paddle and set device to cpu explicitly where available
        try:
            import paddle
            # explicit CPU device selection for Paddle (if available)
            try:
                paddle.set_device('cpu')
            except Exception:
                # older / different installs might not have set_device; ignore safely
                pass
        except Exception as e:
            raise RuntimeError("Paddle not installed. Install paddlepaddle CPU wheel first. Error: " + str(e))
        try:
            from paddleocr import PaddleOCR
        except Exception as e:
            raise RuntimeError("paddleocr not installed: " + str(e))

        # create PaddleOCR with use_gpu=False to force CPU usage
        ocr_instance = PaddleOCR(use_angle_cls=USE_ANGLE, use_gpu=USE_GPU_FLAG, lang="en")
        return ocr_instance

# ---- Normalize PaddleOCR result ----
def parse_paddle(res):
    out = []
    if not res:
        return out
    # handle nested versions
    if isinstance(res, list) and len(res) == 1 and isinstance(res[0], list):
        res = res[0]
    for det in res:
        try:
            box = det[0]
            text_part = det[1]
        except Exception:
            continue
        text = ""
        score = None
        if isinstance(text_part, (list, tuple)):
            if len(text_part) >= 1:
                text = text_part[0] if isinstance(text_part[0], str) else str(text_part[0])
            if len(text_part) >= 2 and isinstance(text_part[1], (float,int)):
                score = float(text_part[1])
        elif isinstance(text_part, str):
            text = text_part
        else:
            text = str(text_part)
        out.append({"text": text, "score": score, "box": [[int(float(p[0])), int(float(p[1]))] for p in box]})
    return out

# ---- OCR with file-backed cache keyed by image mtime ----
def run_ocr_cached():
    mtime = os.path.getmtime(IMAGE_PATH)
    cache = load_cache()
    key = os.path.abspath(IMAGE_PATH)
    entry = cache.get(key)
    if entry and entry.get("mtime") == mtime and "ocr" in entry:
        return entry["ocr"]
    with cache_lock:
        cache = load_cache()
        entry = cache.get(key)
        if entry and entry.get("mtime") == mtime and "ocr" in entry:
            return entry["ocr"]
        ocr = get_ocr()
        img = open_and_downscale(IMAGE_PATH, MAX_DIM)
        try:
            raw = ocr.ocr(img, cls=USE_ANGLE)
        except TypeError:
            raw = ocr.ocr(IMAGE_PATH, cls=USE_ANGLE)
        parsed = parse_paddle(raw)
        cache[key] = {"mtime": mtime, "ocr": parsed, "cached_at": time.time()}
        save_cache(cache)
        return parsed

# ---- Routes ----
@app.route("/")
def index():
    try:
        parsed = run_ocr_cached()
        lines = [it.get("text","") for it in parsed]
        joined = "\n".join(filter(None, lines)) or "(no text found)"
        return render_template_string(HTML_TMPL, img_name=os.path.basename(IMAGE_PATH), text=joined)
    except Exception as e:
        return f"Error running OCR: {e}", 500

@app.route("/image")
def image():
    mime, _ = mimetypes.guess_type(IMAGE_PATH)
    return send_file(IMAGE_PATH, mimetype=mime or "application/octet-stream")

@app.route("/api/ocr")
def api_ocr():
    try:
        parsed = run_ocr_cached()
        return jsonify({"image": os.path.basename(IMAGE_PATH), "ocr": parsed})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health():
    return jsonify({"ok": True, "image": os.path.basename(IMAGE_PATH)})

if __name__ == "__main__":
    print(f"Serving (CPU-only) image: {IMAGE_PATH}")
    print(f"Server: http://{HOST}:{PORT}/")
    app.run(host=HOST, port=PORT, debug=False, threaded=False)
