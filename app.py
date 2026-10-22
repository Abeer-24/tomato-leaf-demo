"""
Tomato leaf disease demo (research / education only).

Runs the trained EfficientNet-B0 with ONNX Runtime, so it fits on a 512 MB free Render instance
(no PyTorch needed). The heat map is Grad-CAM computed in closed form: for a network that ends in
global-average-pooling + one linear layer, Grad-CAM equals ReLU(sum_k W[class, k] * A_k), where A is
the last feature map. The model file exports that feature map next to the logits.

Local test:   py app.py            then open http://127.0.0.1:5000
Render start: gunicorn app:app --workers 1 --threads 2 --timeout 120 --bind 0.0.0.0:$PORT
"""

import base64
import io
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from flask import Flask, render_template_string, request
from PIL import Image, UnidentifiedImageError

HERE = Path(__file__).parent
Image.MAX_IMAGE_PIXELS = 30_000_000
SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
LOW_CONFIDENCE = 0.60

# Numbers shown on the page. Update them if you retrain. They come from the project's own evaluation:
# test split of PlantVillage (in-domain) vs 542 cleaned PlantDoc field images, mean of 3 training runs.
IN_DOMAIN_ACC = "99.7%"
FIELD_ACC = "about 32%"

CLASSES = json.loads((HERE / "classes.json").read_text())
W = np.load(HERE / "classifier.npz")["W"].astype(np.float32)      # (n_classes, 1280)

_opts = ort.SessionOptions()
_opts.intra_op_num_threads = 1
_opts.inter_op_num_threads = 1
SESSION = ort.InferenceSession(str(HERE / "model.onnx"), _opts, providers=["CPUExecutionProvider"])

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 6 * 1024 * 1024


def pretty(name):
    return name.replace("_", " ")


def preprocess(img):
    img = img.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return img, arr.transpose(2, 0, 1)[None].astype(np.float32)


def jet(v):
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)


def to_data_uri(arr01):
    buf = io.BytesIO()
    Image.fromarray((np.clip(arr01, 0, 1) * 255).astype(np.uint8)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def classify(img):
    img224, x = preprocess(img)
    logits, feats = SESSION.run(None, {"image": x})
    logits, feats = logits[0], feats[0]                      # (n_classes,), (1280, 7, 7)
    e = np.exp(logits - logits.max())
    prob = e / e.sum()
    order = np.argsort(-prob)[:3]
    c = int(order[0])
    cam = np.maximum((W[c][:, None, None] * feats).sum(0), 0)
    cam = cam / (cam.max() + 1e-8)
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize((SIZE, SIZE), Image.BICUBIC)
    heat = jet(np.asarray(cam_img, dtype=np.float32) / 255.0)
    base = np.asarray(img224, dtype=np.float32) / 255.0
    return {
        "top3": [(pretty(CLASSES[int(i)]), float(prob[int(i)])) for i in order],
        "low": float(prob[c]) < LOW_CONFIDENCE,
        "original": to_data_uri(base),
        "overlay": to_data_uri(0.55 * base + 0.45 * heat),
        "predicted": pretty(CLASSES[c]),
    }


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tomato leaf disease demo</title>
<style>
 body{font-family:system-ui,Arial,sans-serif;margin:0;background:#f4f6f3;color:#1d2a1d}
 main{max-width:760px;margin:0 auto;padding:18px}
 h1{font-size:1.5rem;margin:.2rem 0 .8rem}
 .warn{background:#fff4d6;border:1px solid #e0b84c;border-radius:8px;padding:10px 14px;font-size:.95rem}
 .err{background:#fde4e4;border:1px solid #d77;border-radius:8px;padding:10px 14px}
 .card{background:#fff;border:1px solid #d9e0d6;border-radius:10px;padding:14px;margin-top:14px}
 form{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-top:14px}
 button{background:#2e7d32;color:#fff;border:0;border-radius:8px;padding:9px 18px;font-size:1rem}
 .imgs{display:flex;gap:12px;flex-wrap:wrap}
 .imgs figure{margin:0;text-align:center;font-size:.85rem}
 .imgs img{width:224px;max-width:100%;border-radius:6px}
 .bar{height:10px;border-radius:5px;background:#2e7d32}
 .row{display:flex;align-items:center;gap:10px;margin:6px 0}
 .row span:first-child{width:170px}
 small,footer{color:#5b665b;font-size:.82rem}
</style></head><body><main>
<h1>Tomato leaf disease demo</h1>
<p class="warn"><strong>Read this first.</strong> This model was trained only on laboratory photos (PlantVillage:
single leaves on plain backgrounds). On that kind of image it is right about {{ in_domain }} of the time. On real
field photos (PlantDoc) the same kind of model was right only {{ field }} of the time, and it was often more than 90%
confident when wrong. It only knows 8 tomato conditions, so any other plant or object still gets a tomato label.
Do not use it to make decisions about real crops.</p>
<form method="post" enctype="multipart/form-data" action="/">
 <input type="file" name="image" accept="image/*" required>
 <button type="submit">Classify</button>
</form>
{% if error %}<p class="err">{{ error }}</p>{% endif %}
{% if result %}
<div class="card">
 <h2 style="margin-top:0">Result: {{ result.predicted }}</h2>
 {% if result.low %}<p class="warn">Low confidence. Treat this answer as a guess.</p>{% endif %}
 {% for name, p in result.top3 %}
  <div class="row"><span>{{ name }}</span>
   <div style="flex:1;background:#e5eae3;border-radius:5px"><div class="bar" style="width:{{ (p*100)|round(1) }}%"></div></div>
   <span>{{ (p*100)|round(1) }}%</span></div>
 {% endfor %}
 <p><small>Confidence is not reliable on field photos. A high percentage does not mean the answer is correct.</small></p>
 <div class="imgs">
  <figure><img src="{{ result.original }}" alt="input"><figcaption>Input (resized to 224&times;224)</figcaption></figure>
  <figure><img src="{{ result.overlay }}" alt="heat map"><figcaption>Where the model looked (Grad-CAM)</figcaption></figure>
 </div>
 <p><small>The heat map is coarse (7&times;7 cells). It shows where the model looked, not why it decided.</small></p>
</div>
{% endif %}
<footer><p>Research and education demo. Trained on PlantVillage (CC BY-NC-SA 4.0, non-commercial).
Uploaded images are processed in memory and not stored.</p></footer>
</main></body></html>"""


def render(result=None, error=None, status=200):
    return render_template_string(PAGE, result=result, error=error,
                                  in_domain=IN_DOMAIN_ACC, field=FIELD_ACC), status


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render()
    file = request.files.get("image")
    if file is None or file.filename == "":
        return render(error="Please choose an image file.", status=400)
    try:
        img = Image.open(file.stream)
        img.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, ValueError):
        return render(error="That file could not be read as an image (JPG or PNG, up to 6 MB).", status=400)
    return render(result=classify(img))


@app.errorhandler(413)
def too_large(_err):
    return render(error="That file is too large. The limit is 6 MB.", status=413)


@app.route("/health")
def health():
    return "ok"


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
