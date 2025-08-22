#!/usr/bin/env python3
"""
Single-file Flask app that:
 - finds an image in the same folder
 - runs PaddleOCR on it (lazy, cached)
 - serves the image and extracted text on a web page
 - provides JSON output at /api/ocr

Requirements:
  pip install flask pillow paddleocr
(Ensure paddleocr/paddlepaddle are installed and working in your env.)
"""

import os
import io
import mimetypes
from typing import List, Dict, Any, Optional
from flask import Flask, send_file, render_template_string, jsonify
from PIL import Image
from paddleocr import PaddleOCR

# ---- Configuration ----
PORT = 8000
HOST = "0.0.0.0"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")
# ------------------------

app = Flask(__name__)

# find an image file in the same directory (first match)
def find_image_file() -> Optional[str]:
    cwd = os.getcwd()
    for fname in os.listdir(cwd):
        if fname.lower().endswith(IMAGE_EXTS) and os.path.isfile(os.path.join(cwd, fname)):
            return os.path.join(cwd, fname)
    return None

IMAGE_PATH = find_image_file()
if not IMAGE_PATH:
    raise SystemExit("No image found in current directory. Place an image (jpg/png/etc.) in this folder and re-run.")

# Initialize OCR object lazily to avoid long startup cost before first request
ocr_instance: Optional[PaddleOCR] = None
ocr_cache: Optional[List[Dict[str, Any]]] = None  # cached parsed OCR result


def get_ocr_instance() -> PaddleOCR:
    global ocr_instance
    if ocr_instance is None:
        # initialize PaddleOCR; tweak args as needed for performance
        # (use_angle_cls True helps rotated text detection; set lang to 'en' or as needed)
        ocr_instance = PaddleOCR(lang="en", use_angle_cls=True)
    return ocr_instance


def parse_paddle_result(res) -> List[Dict[str, Any]]:
    """
    Normalize various PaddleOCR return formats into a list of dicts:
      [{ 'text': str, 'score': float|None, 'box': [[x,y],...] }, ...]
    """
    if not res:
        return []

    # PaddleOCR historically returns either:
    #  - a list of detections (each detection: [box, (text, score)])
    #  - or a list containing one list: [ detections ]
    # We'll normalize both shapes.
    detections = res
    if isinstance(res, list) and len(res) == 1 and isinstance(res[0], list):
        detections = res[0]

    output = []
    for det in detections:
        # det expected to be like [box, (text, score)] or (box, [text,score])
        try:
            box = det[0]
            text_part = det[1]
        except Exception:
            # Try alternative nested shapes
            # Some versions return list of [ [box, (text,score)], ... ] inside extra nesting
            continue

        # normalize box to list of point tuples/lists
        norm_box = []
        try:
            for p in box:
                # some boxes are nested as floats; convert to int for display
                norm_box.append([int(float(p[0])), int(float(p[1]))])
        except Exception:
            norm_box = box  # fallback: keep whatever it is

        # extract text and score heuristically
        text = None
        score = None
        if isinstance(text_part, (list, tuple)):
            # text_part might be ('text', 0.98) or ['text', 0.98]
            for v in text_part:
                if isinstance(v, str):
                    text = v
                    break
            for v in text_part:
                if isinstance(v, (float, int)):
                    score = float(v)
                    break
        elif isinstance(text_part, str):
            text = text_part
        else:
            # some variants: det[1][0] is text
            try:
                cand = text_part[0]
                if isinstance(cand, str):
                    text = cand
            except Exception:
                pass

        if text is None:
            # final fallback: try converting the whole part
            text = str(text_part)

        output.append({"text": text, "score": score, "box": norm_box})

    return output


def run_ocr_and_cache() -> List[Dict[str, Any]]:
    global ocr_cache
    if ocr_cache is not None:
        return ocr_cache
    ocr = get_ocr_instance()
    # PaddleOCR accepts either image path or PIL image; pass path for simplicity
    try:
        raw = ocr.ocr(IMAGE_PATH, cls=True)
    except TypeError:
        # some versions expect image object
        img = Image.open(IMAGE_PATH).convert("RGB")
        raw = ocr.ocr(img, cls=True)
    parsed = parse_paddle_result(raw)
    ocr_cache = parsed
    return parsed


# Flask routes
HTML_TMPL = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>OCR Host</title>
  <style>
    body{font-family:system-ui,Segoe UI,Roboto,Helvetica,Arial; margin:24px;}
    .container{display:flex; gap:24px; align-items:flex-start;}
    .imgbox img{max-width:600px; height:auto; border:1px solid #ddd; padding:4px; background:#fafafa;}
    .result{white-space:pre-wrap; font-family:monospace; max-width:60ch; background:#f7f7f7;padding:12px;border-radius:6px;}
    h1{margin-bottom:6px}
  </style>
</head>
<body>
  <h1>OCR Host</h1>
  <p>Image: <strong>{{ img_name }}</strong></p>
  <div class="container">
    <div class="imgbox">
      <img src="/image" alt="image">
      <p><a href="/image" download>Download image</a></p>
    </div>
    <div>
      <h2>Extracted text</h2>
      <div class="result">{{ text }}</div>
      <p><a href="/api/ocr">Get JSON output</a></p>
    </div>
  </div>
</body>
</html>
"""

@app.route("/")
def index():
    try:
        parsed = run_ocr_and_cache()
        # join text lines for display
        text_lines = [item.get("text", "") for item in parsed]
        joined = "\n".join(text_lines) if text_lines else "(no text found)"
        return render_template_string(HTML_TMPL, img_name=os.path.basename(IMAGE_PATH), text=joined)
    except Exception as e:
        return f"Error running OCR: {e}", 500

@app.route("/image")
def image():
    # serve the image file directly with correct mimetype
    mime, _ = mimetypes.guess_type(IMAGE_PATH)
    mime = mime or "application/octet-stream"
    return send_file(IMAGE_PATH, mimetype=mime)

@app.route("/api/ocr")
def api_ocr():
    try:
        parsed = run_ocr_and_cache()
        return jsonify({"image": os.path.basename(IMAGE_PATH), "ocr": parsed})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print(f"Serving image: {IMAGE_PATH}")
    print(f"Start server at http://{HOST}:{PORT}/")
    # debug=False to avoid multiple Paddle initializations in dev servers
    app.run(host=HOST, port=PORT, debug=False)
