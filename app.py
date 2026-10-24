"""
Tomato leaf check: a research and education demo.

ONNX Runtime + Flask, sized for a 512 MB free Render instance (no PyTorch). The heat map is Grad-CAM in
closed form: for a network that ends in global-average-pooling plus one linear layer, Grad-CAM equals
ReLU(sum_k W[class, k] * A_k), where A is the last feature map. model.onnx outputs that map next to the logits.

Local test:   py app.py            then open http://127.0.0.1:5000
Render start: gunicorn app:app --workers 1 --threads 2 --timeout 120 --bind 0.0.0.0:$PORT
"""

import base64
import io
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from flask import Flask, jsonify, request
from PIL import Image, UnidentifiedImageError

HERE = Path(__file__).parent
Image.MAX_IMAGE_PIXELS = 30_000_000
SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
LOW_CONFIDENCE = 0.60

# Numbers shown on the page. Update them if you retrain. They come from the project's own evaluation:
# PlantVillage test split (in-domain) vs 542 cleaned PlantDoc field images, mean of 3 training runs.
IN_DOMAIN_ACC = "99.7%"
FIELD_ACC = "about 32%"

LATIN = {
    "Bacterial_spot": "Xanthomonas species",
    "Early_blight": "Alternaria solani",
    "Late_blight": "Phytophthora infestans",
    "Leaf_Mold": "Passalora fulva",
    "Mosaic_virus": "Tomato mosaic virus",
    "Septoria_leaf_spot": "Septoria lycopersici",
    "Yellow_leaf_curl": "Tomato yellow leaf curl virus",
    "healthy": "",
}

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


def heat_png(cam01):
    """Transparent heat map: colour from jet, opacity follows the activation, so quiet areas stay clear."""
    rgba = np.dstack([jet(cam01), np.clip(cam01, 0, 1) ** 0.7])
    buf = io.BytesIO()
    Image.fromarray((rgba * 255).astype(np.uint8), "RGBA").save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def classify(img):
    _img224, x = preprocess(img)
    logits, feats = SESSION.run(None, {"image": x})
    logits, feats = logits[0], feats[0]                      # (n_classes,), (1280, 7, 7)
    e = np.exp(logits - logits.max())
    prob = e / e.sum()
    order = np.argsort(-prob)[:3]
    c = int(order[0])
    cam = np.maximum((W[c][:, None, None] * feats).sum(0), 0)
    cam = cam / (cam.max() + 1e-8)
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize((SIZE, SIZE), Image.BICUBIC)
    return {
        "key": CLASSES[c],
        "predicted": pretty(CLASSES[c]),
        "latin": LATIN.get(CLASSES[c], ""),
        "low": float(prob[c]) < LOW_CONFIDENCE,
        "top3": [{"name": pretty(CLASSES[int(i)]), "p": float(prob[int(i)])} for i in order],
        "heat": heat_png(np.asarray(cam_img, dtype=np.float32) / 255.0),
    }


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>Tomato leaf check</title>
<style>
:root{
  --page:#E4EBE6; --plate:#FAFBF9; --ink:#14302A; --muted:#4B605A; --line:rgba(20,48,42,.28);
  --tomato:#C93B25; --on-tomato:#FFFFFF; --caution-bg:#F3E2A0; --caution-ink:#4A3A00;
  --serif:"Iowan Old Style","Palatino Linotype",Palatino,"Book Antiqua",Georgia,serif;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
}
@media (prefers-color-scheme:dark){
  :root{--page:#101C18; --plate:#182822; --ink:#E4EEE9; --muted:#A3B8B0; --line:rgba(228,238,233,.30);
        --tomato:#E5604A; --on-tomato:#1A0904; --caution-bg:#3B3416; --caution-ink:#F2DE92;}
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--page);color:var(--ink);font:16px/1.5 var(--sans)}
.notice{background:var(--caution-bg);color:var(--caution-ink);padding:10px 16px;font-size:.92rem}
.notice>*{max-width:62rem;margin-left:auto;margin-right:auto}
.notice p{margin-top:0;margin-bottom:0}
.notice details{margin-top:2px}
.notice summary{cursor:pointer;font-weight:600;padding:8px 0}
.notice details p{padding-bottom:6px}
main{max-width:62rem;margin:0 auto;padding:22px 16px 44px}
h1{font:600 1.9rem/1.15 var(--serif);margin:0 0 .35rem}
.lede{color:var(--muted);max-width:38rem;margin:0}
.stage{display:grid;gap:22px;margin-top:22px}
@media (min-width:820px){.stage{grid-template-columns:minmax(0,1.05fr) minmax(0,1fr);align-items:start}}

.plate{position:relative;background:var(--plate);border:1px solid var(--line);padding:26px;display:grid;place-items:center;min-height:320px}
.corner{position:absolute;width:16px;height:16px;border:2px solid var(--ink);opacity:.7;pointer-events:none}
.corner.tl{top:8px;left:8px;border-right:0;border-bottom:0}
.corner.tr{top:8px;right:8px;border-left:0;border-bottom:0}
.corner.bl{bottom:8px;left:8px;border-right:0;border-top:0}
.corner.br{bottom:8px;right:8px;border-left:0;border-top:0}
.plate.over{outline:2px dashed var(--tomato);outline-offset:-10px}
.drop{text-align:center;max-width:22rem}
.drop-title{font:600 1.3rem/1.25 var(--serif);margin:0 0 .4rem}
.drop-sub{color:var(--muted);font-size:.92rem;margin:0 0 1.1rem}
.frame{position:relative;width:min(100%,460px,calc(62vh * var(--ar,1)));aspect-ratio:var(--ar,1)}
.frame img{position:absolute;inset:0;width:100%;height:100%;display:block}
#heat{transition:opacity .15s}

.side{display:grid;gap:16px;align-content:start}
.row{display:flex;gap:10px;flex-wrap:wrap;justify-content:center}
.side .row{justify-content:flex-start}
.btn{appearance:none;border:1px solid var(--ink);background:transparent;color:var(--ink);border-radius:4px;min-height:44px;padding:0 18px;font:600 1rem/1 var(--sans);cursor:pointer}
.btn.primary{background:var(--tomato);border-color:var(--tomato);color:var(--on-tomato)}
.btn:disabled{opacity:.5;cursor:not-allowed}
.btn:focus-visible,summary:focus-visible,input[type=range]:focus-visible{outline:3px solid var(--tomato);outline-offset:3px}
.spin{display:inline-block;width:.95em;height:.95em;margin-right:.5em;vertical-align:-.12em;border:2px solid currentColor;border-right-color:transparent;border-radius:50%;animation:spin .8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.status{margin:0;color:var(--muted);font-size:.92rem;min-height:1.4em}
.status.error{color:var(--tomato);font-weight:600}

.slip{border:1px solid var(--ink);outline:1px solid var(--ink);outline-offset:-6px;padding:22px 24px;background:var(--plate)}
.slip .k{margin:0;color:var(--muted);font-size:.92rem}
.slip .name{font:600 2.15rem/1.1 var(--serif);margin:.15rem 0 .15rem;text-transform:capitalize}
.slip .latin{font:italic 1.05rem/1.35 var(--serif);color:var(--muted);margin:0;min-height:1.35em}
.chip{display:inline-block;margin:.85rem 0 0;padding:.28rem .6rem;border:1px solid var(--ink);border-radius:3px;font-size:.88rem}
.chip.low{background:var(--caution-bg);color:var(--caution-ink);border-color:var(--caution-ink);font-weight:600}
.cand{margin:1.1rem 0 0;padding:0;list-style:none;display:grid;gap:.6rem}
.cand li{display:grid;grid-template-columns:minmax(0,9.5rem) 1fr 3rem;gap:10px;align-items:center;font-size:.95rem}
.cand .nm{overflow-wrap:anywhere}
.meter{height:6px;background:var(--line);border-radius:3px;overflow:hidden}
.meter span{display:block;height:100%;background:var(--muted)}
.cand li:first-child .meter span{background:var(--tomato)}
.cand .pct{text-align:right;font-variant-numeric:tabular-nums}
.fine{color:var(--muted);font-size:.85rem;margin:.9rem 0 0}
.heatctl label{display:block;font-weight:600;margin-bottom:.2rem}
.heatctl input{width:100%;accent-color:var(--tomato);min-height:32px;margin:0}
.heatctl .fine{margin-top:.2rem}
footer{margin-top:30px;color:var(--muted);font-size:.85rem;max-width:44rem}
.sr{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap}
[hidden]{display:none !important}
@media (prefers-reduced-motion:reduce){.spin{animation:none;border-right-color:currentColor}#heat{transition:none}}
</style>
</head>
<body>
<div class="notice" role="note">
  <p><strong>Research demo.</strong> The model learned from lab photos, so it is unreliable on photos taken in a garden or field. Never use it to decide what to do with a real crop.</p>
  <details>
    <summary>Why it is unreliable</summary>
    <p>On lab-style test images it is right __IN_DOMAIN__ of the time. On 542 real field photos from PlantDoc, similar models were right __FIELD__ of the time, and were often more than 90% sure when wrong. It knows only these eight: bacterial spot, early blight, late blight, leaf mold, mosaic virus, septoria leaf spot, yellow leaf curl virus and healthy. Any other plant or object still gets one of them.</p>
  </details>
</div>
<noscript><p class="notice">This page needs JavaScript to shrink, upload and check a photo.</p></noscript>
<main>
  <h1>Tomato leaf check</h1>
  <p class="lede">Photograph one tomato leaf. The model names the closest of the 8 conditions it knows and shows where it looked.</p>

  <div class="stage">
    <div class="plate" id="plate">
      <span class="corner tl"></span><span class="corner tr"></span><span class="corner bl"></span><span class="corner br"></span>
      <div class="drop" id="drop">
        <p class="drop-title">Drop a leaf photo here</p>
        <p class="drop-sub">One leaf filling the frame on a plain background is closest to what the model learned from.</p>
        <div class="row">
          <button type="button" class="btn primary" id="snap">Take photo</button>
          <button type="button" class="btn" id="pick">Choose file</button>
        </div>
      </div>
      <div class="frame" id="frame" hidden>
        <img id="photo" alt="Your leaf photo">
        <img id="heat" alt="Heat map over your photo" hidden>
      </div>
    </div>

    <div class="side">
      <div class="row">
        <button type="button" class="btn primary" id="check" disabled>Check leaf</button>
        <button type="button" class="btn" id="reset" hidden>Choose another photo</button>
      </div>
      <p class="status" id="status" role="status">Add a photo to begin.</p>

      <section class="slip" id="slip" hidden aria-labelledby="name">
        <p class="k">Closest match</p>
        <h2 class="name" id="name"></h2>
        <p class="latin" id="latin"></p>
        <span class="chip" id="chip"></span>
        <ul class="cand" id="cand" aria-label="Other candidates"></ul>
        <p class="fine">Percentages are the model's own confidence. On field photos they are not reliable: a high number does not mean the answer is right.</p>
      </section>

      <div class="heatctl" id="heatctl" hidden>
        <label for="heatRange">Heat map</label>
        <input type="range" id="heatRange" min="0" max="100" value="70">
        <p class="fine">Warmer colors mark areas that pushed the model toward this answer. The map is coarse (7 by 7 cells) and shows where it looked, not why it decided.</p>
      </div>
    </div>
  </div>

  <footer>
    Built for a college research project on how well lab-trained models transfer to field photos.
    Trained on PlantVillage (CC BY-NC-SA 4.0, non-commercial use only). Your photo is shrunk in your browser,
    checked in memory and not stored.
  </footer>
  <div class="sr" id="live" aria-live="polite"></div>
  <input type="file" id="file" accept="image/*" class="sr" tabindex="-1" aria-hidden="true">
  <input type="file" id="cam" accept="image/*" capture="environment" class="sr" tabindex="-1" aria-hidden="true">
</main>
<script>
(function () {
  var MAX_SIDE = 1024;
  var $ = function (id) { return document.getElementById(id); };
  var el = {};
  ['plate','drop','frame','photo','heat','file','cam','pick','snap','check','reset','status','slip','name',
   'latin','chip','cand','heatctl','heatRange','live'].forEach(function (id) { el[id] = $(id); });
  var prepared = null, slowTimer = null, controller = null;

  function setStatus(msg, isError) {
    el.status.textContent = msg;
    el.status.className = 'status' + (isError ? ' error' : '');
  }
  function loadImage(url) {
    return new Promise(function (resolve, reject) {
      var img = new Image();
      img.onload = function () { resolve(img); };
      img.onerror = reject;
      img.src = url;
    });
  }
  function prepare(file) {
    if (file.type && file.type.indexOf('image/') !== 0) {
      return Promise.reject(new Error('That is not an image. Choose a JPG or PNG photo.'));
    }
    var url = URL.createObjectURL(file);
    return loadImage(url).then(function (img) {
      var s = Math.min(1, MAX_SIDE / Math.max(img.naturalWidth, img.naturalHeight));
      var w = Math.max(1, Math.round(img.naturalWidth * s));
      var h = Math.max(1, Math.round(img.naturalHeight * s));
      var canvas = document.createElement('canvas');
      canvas.width = w; canvas.height = h;
      canvas.getContext('2d').drawImage(img, 0, 0, w, h);
      return new Promise(function (resolve) {
        canvas.toBlob(function (blob) { resolve({ url: url, blob: blob || file, w: w, h: h }); }, 'image/jpeg', 0.9);
      });
    }, function () {
      URL.revokeObjectURL(url);
      throw new Error("This browser can't open that photo format. Use a JPG or PNG.");
    });
  }
  function clearResult() {
    el.slip.hidden = true; el.heatctl.hidden = true; el.heat.hidden = true; el.heat.removeAttribute('src');
  }
  function showPhoto(p) {
    el.photo.src = p.url;
    el.frame.style.setProperty('--ar', (p.w / p.h).toFixed(4));
    el.drop.hidden = true; el.frame.hidden = false;
    el.check.disabled = false; el.reset.hidden = false;
    clearResult();
    setStatus('Photo ready. Check it when you are ready.');
  }
  function resetAll() {
    if (controller) controller.abort();
    if (prepared) URL.revokeObjectURL(prepared.url);
    prepared = null;
    clearResult();
    el.frame.hidden = true; el.drop.hidden = false;
    el.check.disabled = true; el.reset.hidden = true;
    el.file.value = ''; el.cam.value = '';
    busy(false);
    setStatus('Add a photo to begin.');
  }
  function busy(on) {
    if (on) {
      el.check.disabled = true;
      el.check.innerHTML = '<span class="spin" aria-hidden="true"></span>Checking';
    } else {
      el.check.textContent = 'Check leaf';
      el.check.disabled = !prepared;
      clearTimeout(slowTimer);
    }
  }
  function handleFile(file) {
    if (!file) return;
    setStatus('Preparing your photo');
    prepare(file).then(function (p) {
      if (prepared) URL.revokeObjectURL(prepared.url);
      prepared = p; showPhoto(p);
    }).catch(function (err) { setStatus(err.message, true); });
  }
  function showResult(data) {
    el.name.textContent = data.predicted;
    el.latin.textContent = data.latin || (data.key === 'healthy'
      ? 'Closest to its healthy examples. This is not a health check.' : '');
    el.chip.textContent = data.low ? 'Low confidence: treat this as a guess' : 'Confidence is not reliable on field photos';
    el.chip.className = 'chip' + (data.low ? ' low' : '');
    el.cand.textContent = '';
    data.top3.forEach(function (c) {
      var li = document.createElement('li');
      var nm = document.createElement('span'); nm.className = 'nm'; nm.textContent = c.name;
      var meter = document.createElement('div'); meter.className = 'meter';
      var bar = document.createElement('span'); bar.style.width = Math.max(1, Math.round(c.p * 100)) + '%';
      meter.appendChild(bar);
      var pct = document.createElement('span'); pct.className = 'pct'; pct.textContent = Math.round(c.p * 100) + '%';
      li.appendChild(nm); li.appendChild(meter); li.appendChild(pct);
      el.cand.appendChild(li);
    });
    el.heat.src = data.heat;
    el.heat.style.opacity = el.heatRange.value / 100;
    el.heat.hidden = false; el.heatctl.hidden = false; el.slip.hidden = false;
    el.live.textContent = 'Closest match: ' + data.predicted + (data.low ? '. Low confidence.' : '.');
    setStatus('Done. Choose another photo to try again.');
  }
  function check() {
    if (!prepared) return;
    busy(true); clearResult(); setStatus('Checking your leaf');
    slowTimer = setTimeout(function () {
      setStatus('Still working. The free server can take up to a minute to wake up.');
    }, 6000);
    controller = new AbortController();
    var abortTimer = setTimeout(function () { controller.abort(); }, 120000);
    var fd = new FormData();
    fd.append('image', prepared.blob, 'leaf.jpg');
    fetch('/api/classify', { method: 'POST', body: fd, signal: controller.signal })
      .then(function (res) {
        return res.json().catch(function () { return {}; }).then(function (data) {
          if (!res.ok) throw new Error(data.error || 'The server could not check that photo.');
          return data;
        });
      })
      .then(showResult)
      .catch(function (err) {
        setStatus(err.name === 'AbortError'
          ? "That took too long. The free server may be waking up: try again in a minute."
          : (err.message === 'Failed to fetch'
              ? "Could not reach the server. Check your connection and try again."
              : err.message), true);
      })
      .then(function () { clearTimeout(abortTimer); busy(false); });
  }

  el.snap.addEventListener('click', function () { el.cam.click(); });
  el.pick.addEventListener('click', function () { el.file.click(); });
  el.file.addEventListener('change', function () { handleFile(el.file.files[0]); });
  el.cam.addEventListener('change', function () { handleFile(el.cam.files[0]); });
  el.check.addEventListener('click', check);
  el.reset.addEventListener('click', resetAll);
  el.heatRange.addEventListener('input', function () { el.heat.style.opacity = el.heatRange.value / 100; });
  ['dragenter', 'dragover'].forEach(function (t) {
    el.plate.addEventListener(t, function (e) { e.preventDefault(); el.plate.classList.add('over'); });
  });
  ['dragleave', 'drop'].forEach(function (t) {
    el.plate.addEventListener(t, function (e) { e.preventDefault(); el.plate.classList.remove('over'); });
  });
  el.plate.addEventListener('drop', function (e) { handleFile(e.dataTransfer.files[0]); });
})();
</script>
</body>
</html>"""


def page():
    return PAGE.replace("__IN_DOMAIN__", IN_DOMAIN_ACC).replace("__FIELD__", FIELD_ACC)


@app.route("/")
def index():
    return page()


@app.route("/api/classify", methods=["POST"])
def api_classify():
    file = request.files.get("image")
    if file is None or file.filename == "":
        return jsonify(error="No photo was received. Add a photo and try again."), 400
    try:
        img = Image.open(file.stream)
        img.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, ValueError):
        return jsonify(error="The server could not read that file. Use a JPG or PNG photo."), 400
    resp = jsonify(classify(img))
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.errorhandler(413)
def too_large(_err):
    return jsonify(error="That photo is larger than 6 MB. Choose a smaller one."), 413


@app.route("/health")
def health():
    return "ok"


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
