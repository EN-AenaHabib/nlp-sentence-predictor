"""
NLP Sentence Completion & Next Word Predictor
=============================================
Deployed on Hugging Face Spaces (Flask)
Dataset: WikiText-2 (via HuggingFace datasets) — much richer than Brown Corpus
Models : Bigram & Trigram with Laplace smoothing + UNK handling
"""

import os, re, math, json
from collections import defaultdict, Counter
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# ── 1. Load & preprocess WikiText-2 ──────────────────────────────────────────
print("Loading WikiText-2 dataset…")

try:
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    raw_text = "\n".join(ds["text"])
except Exception as e:
    print(f"datasets lib failed ({e}), falling back to NLTK Brown corpus…")
    import nltk
    nltk.download("brown", quiet=True)
    from nltk.corpus import brown
    raw_text = " ".join(" ".join(s) for s in brown.sents())

# Sentence-split on periods / newlines
sentences_raw = re.split(r'(?<=[.!?])\s+|\n+', raw_text)

def preprocess(sentence):
    words = re.findall(r"[a-z']+", sentence.lower())
    return [w for w in words if len(w) > 0]

clean_sents = [preprocess(s) for s in sentences_raw if len(preprocess(s)) >= 3]
print(f"Total usable sentences: {len(clean_sents):,}")

# ── 2. Vocabulary ─────────────────────────────────────────────────────────────
def build_vocab(sentences, min_freq=3):
    counts = Counter(w for s in sentences for w in s)
    vocab = {w for w, f in counts.items() if f >= min_freq}
    vocab.update(["<UNK>", "<s>", "</s>"])
    return vocab

split = int(len(clean_sents) * 0.9)
train_sents = clean_sents[:split]

vocab = build_vocab(train_sents, min_freq=3)
print(f"Vocabulary size: {len(vocab):,}")

def unk_sent(sentence, vocab):
    return [w if w in vocab else "<UNK>" for w in sentence]

train_data = [unk_sent(s, vocab) for s in train_sents]

# ── 3. Build models ───────────────────────────────────────────────────────────
def build_bigram(data):
    model = defaultdict(Counter)
    for words in data:
        padded = ["<s>"] + words + ["</s>"]
        for i in range(len(padded) - 1):
            model[padded[i]][padded[i+1]] += 1
    return model

def build_trigram(data):
    model = defaultdict(Counter)
    for words in data:
        padded = ["<s>", "<s>"] + words + ["</s>"]
        for i in range(len(padded) - 2):
            model[(padded[i], padded[i+1])][padded[i+2]] += 1
    return model

print("Building bigram model…")
bigram_model = build_bigram(train_data)
print("Building trigram model…")
trigram_model = build_trigram(train_data)
print("Models ready!")

VOCAB_SIZE = len(vocab)
SKIP = {"<UNK>", "<s>", "</s>"}

# ── 4. Prediction helpers ─────────────────────────────────────────────────────
def unk_word(w):
    return w if w in vocab else "<UNK>"

def predict_bigram(word, top_k=5):
    word = unk_word(word.lower())
    dist = bigram_model[word]
    total = sum(dist.values())
    scored = {
        w: (dist.get(w, 0) + 1) / (total + VOCAB_SIZE)
        for w in dist if w not in SKIP
    }
    # Also consider top vocab words unseen in context
    if not scored:
        scored = {"the": 1/VOCAB_SIZE, "a": 1/VOCAB_SIZE,
                  "in": 1/VOCAB_SIZE, "of": 1/VOCAB_SIZE, "and": 1/VOCAB_SIZE}
    return sorted(scored.items(), key=lambda x: x[1], reverse=True)[:top_k]

def predict_trigram(w1, w2, top_k=5):
    w1, w2 = unk_word(w1.lower()), unk_word(w2.lower())
    dist = trigram_model[(w1, w2)]
    total = sum(dist.values())
    if total == 0:
        # Backoff to bigram
        return predict_bigram(w2, top_k)
    scored = {
        w: (dist.get(w, 0) + 1) / (total + VOCAB_SIZE)
        for w in dist if w not in SKIP
    }
    return sorted(scored.items(), key=lambda x: x[1], reverse=True)[:top_k]

def complete_sentence(seed_text, model_type="trigram", max_words=10):
    words = [w for w in preprocess(seed_text) if w]
    if not words:
        return seed_text, []
    words_unk = [unk_word(w) for w in words]
    generated = list(words_unk)

    for _ in range(max_words):
        if model_type == "trigram" and len(generated) >= 2:
            preds = predict_trigram(generated[-2], generated[-1], top_k=1)
        else:
            preds = predict_bigram(generated[-1], top_k=1)
        if not preds:
            break
        next_word = preds[0][0]
        if next_word in SKIP:
            break
        generated.append(next_word)

    # Replace UNK tokens with original words where possible
    result = []
    for i, w in enumerate(generated):
        if w == "<UNK>" and i < len(words):
            result.append(words[i])
        else:
            result.append(w if w != "<UNK>" else "")

    result = [w for w in result if w]
    new_words = result[len(words):]
    return " ".join(result), new_words

# ── 5. Flask routes ───────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(HTML_PAGE)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "").strip()
    mode = data.get("mode", "trigram")
    if not text:
        return jsonify([])
    words = preprocess(text)
    if not words:
        return jsonify([])
    if mode == "trigram" and len(words) >= 2:
        preds = predict_trigram(words[-2], words[-1], top_k=5)
    else:
        preds = predict_bigram(words[-1], top_k=5)
    result = [{"word": w, "prob": round(p * 100, 4)} for w, p in preds]
    return jsonify(result)

@app.route("/complete", methods=["POST"])
def complete():
    data = request.get_json()
    text = data.get("text", "").strip()
    mode = data.get("mode", "trigram")
    max_words = int(data.get("max_words", 10))
    if not text:
        return jsonify({"full": "", "added": []})
    full, added = complete_sentence(text, model_type=mode, max_words=max_words)
    return jsonify({"full": full, "added": added})

@app.route("/health")
def health():
    return jsonify({"status": "ok", "vocab_size": VOCAB_SIZE,
                    "bigram_contexts": len(bigram_model),
                    "trigram_contexts": len(trigram_model)})

# ── 6. HTML Page (full frontend) ──────────────────────────────────────────────
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NLP Sentence Predictor · NUTECH AI-23</title>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=JetBrains+Mono:wght@300;400;600&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}

:root{
  --bg:#080a0f;
  --surface:#0f1218;
  --surface2:#161b24;
  --border:#1e2535;
  --border2:#252d3d;
  --accent1:#00ffb3;
  --accent2:#ff6eb4;
  --accent3:#4db8ff;
  --accent4:#ffd166;
  --text:#dde4f0;
  --muted:#4a5568;
  --muted2:#6b7a99;
  --font-sans:'Syne',sans-serif;
  --font-mono:'JetBrains Mono',monospace;
  --glow1:rgba(0,255,179,.15);
  --glow2:rgba(255,110,180,.15);
  --glow3:rgba(77,184,255,.15);
}

body{
  font-family:var(--font-sans);
  background:var(--bg);
  color:var(--text);
  min-height:100vh;
  overflow-x:hidden;
}

/* ── Noise texture overlay ── */
body::before{
  content:'';
  position:fixed;inset:0;
  background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.03'/%3E%3C/svg%3E");
  pointer-events:none;z-index:0;opacity:.4;
}

/* ── Grid background ── */
body::after{
  content:'';
  position:fixed;inset:0;
  background-image:
    linear-gradient(rgba(0,255,179,.03) 1px,transparent 1px),
    linear-gradient(90deg,rgba(0,255,179,.03) 1px,transparent 1px);
  background-size:40px 40px;
  pointer-events:none;z-index:0;
}

.wrap{
  position:relative;z-index:1;
  max-width:900px;
  margin:0 auto;
  padding:40px 20px 80px;
}

/* ── Header ── */
.header{text-align:center;margin-bottom:48px;}

.header-eyebrow{
  display:inline-flex;align-items:center;gap:8px;
  font-family:var(--font-mono);font-size:10px;
  letter-spacing:4px;text-transform:uppercase;
  color:var(--accent1);
  border:1px solid rgba(0,255,179,.3);
  padding:5px 14px;border-radius:99px;
  margin-bottom:20px;
  animation:fadeUp .6s ease both;
}
.header-eyebrow .dot{
  width:6px;height:6px;border-radius:50%;
  background:var(--accent1);
  animation:pulse 2s ease infinite;
}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.4;transform:scale(.7)}}

.header h1{
  font-size:clamp(28px,6vw,52px);
  font-weight:800;letter-spacing:-2px;line-height:1.05;
  background:linear-gradient(135deg,var(--accent1) 0%,var(--accent3) 45%,var(--accent2) 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  background-clip:text;
  animation:fadeUp .7s .1s ease both;
}

.header .sub{
  margin-top:12px;
  font-family:var(--font-mono);font-size:13px;
  color:var(--muted2);letter-spacing:.5px;
  animation:fadeUp .7s .2s ease both;
}

.header .tags{
  display:flex;gap:8px;justify-content:center;flex-wrap:wrap;
  margin-top:16px;
  animation:fadeUp .7s .3s ease both;
}
.tag{
  font-family:var(--font-mono);font-size:10px;
  letter-spacing:2px;text-transform:uppercase;
  padding:3px 10px;border-radius:4px;
}
.tag-green{background:rgba(0,255,179,.1);color:var(--accent1);border:1px solid rgba(0,255,179,.2);}
.tag-blue {background:rgba(77,184,255,.1);color:var(--accent3);border:1px solid rgba(77,184,255,.2);}
.tag-pink {background:rgba(255,110,180,.1);color:var(--accent2);border:1px solid rgba(255,110,180,.2);}
.tag-gold {background:rgba(255,209,102,.1);color:var(--accent4);border:1px solid rgba(255,209,102,.2);}

@keyframes fadeUp{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:none}}

/* ── Stats bar ── */
.stats-bar{
  display:flex;gap:1px;
  background:var(--border);
  border-radius:12px;overflow:hidden;
  margin-bottom:28px;
  border:1px solid var(--border);
  animation:fadeUp .7s .4s ease both;
}
.stat{
  flex:1;padding:14px 16px;
  background:var(--surface);
  text-align:center;
}
.stat-val{
  font-family:var(--font-mono);font-size:18px;
  font-weight:600;color:var(--accent1);
  display:block;
}
.stat-lbl{
  font-family:var(--font-mono);font-size:9px;
  letter-spacing:2px;text-transform:uppercase;
  color:var(--muted);margin-top:2px;display:block;
}

/* ── Card ── */
.card{
  background:var(--surface);
  border:1px solid var(--border2);
  border-radius:16px;
  padding:28px;
  margin-bottom:20px;
  position:relative;overflow:hidden;
  animation:fadeUp .7s .5s ease both;
}
.card::before{
  content:'';position:absolute;
  top:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent,var(--accent1),var(--accent3),var(--accent2),transparent);
  opacity:.6;
}

/* ── Section label ── */
.slabel{
  font-family:var(--font-mono);font-size:9px;
  letter-spacing:3px;text-transform:uppercase;
  color:var(--muted);margin-bottom:14px;
  display:flex;align-items:center;gap:10px;
}
.slabel::after{content:'';flex:1;height:1px;background:var(--border2);}

/* ── Model controls ── */
.controls{
  display:flex;gap:12px;flex-wrap:wrap;
  align-items:center;margin-bottom:18px;
}

.seg{
  display:flex;
  border:1px solid var(--border2);
  border-radius:8px;overflow:hidden;
}
.seg-btn{
  background:transparent;border:none;
  color:var(--muted2);
  font-family:var(--font-mono);font-size:11px;
  letter-spacing:1px;
  padding:8px 18px;cursor:pointer;
  transition:all .2s;
  position:relative;
}
.seg-btn+.seg-btn{border-left:1px solid var(--border2);}
.seg-btn.active{background:var(--accent1);color:#080a0f;font-weight:600;}

.range-wrap{
  display:flex;align-items:center;gap:10px;
  font-family:var(--font-mono);font-size:11px;color:var(--muted2);
}
input[type=range]{
  -webkit-appearance:none;appearance:none;
  height:4px;border-radius:99px;
  background:var(--border2);outline:none;cursor:pointer;width:100px;
}
input[type=range]::-webkit-slider-thumb{
  -webkit-appearance:none;
  width:14px;height:14px;border-radius:50%;
  background:var(--accent1);cursor:pointer;
}
.range-val{
  color:var(--accent1);font-weight:600;
  min-width:20px;text-align:center;
}

/* ── Textarea ── */
.input-wrap{position:relative;}
textarea{
  width:100%;min-height:100px;
  background:var(--surface2);
  border:1px solid var(--border2);
  border-radius:12px;
  color:var(--text);
  font-family:var(--font-mono);font-size:15px;
  line-height:1.6;
  padding:16px 18px 36px;
  resize:vertical;outline:none;
  transition:border-color .25s,box-shadow .25s;
}
textarea:focus{
  border-color:var(--accent1);
  box-shadow:0 0 0 3px var(--glow1),0 0 20px var(--glow1);
}
textarea::placeholder{color:var(--muted);}

.input-footer{
  position:absolute;bottom:10px;left:16px;right:16px;
  display:flex;justify-content:space-between;
  pointer-events:none;
}
.input-footer span{
  font-family:var(--font-mono);font-size:10px;
  color:var(--muted);
}

/* ── Buttons ── */
.action-row{display:flex;gap:10px;margin-top:14px;flex-wrap:wrap;}

.btn{
  display:inline-flex;align-items:center;gap:8px;
  padding:11px 24px;border:none;border-radius:8px;
  font-family:var(--font-sans);font-size:13px;font-weight:700;
  cursor:pointer;transition:all .2s;letter-spacing:.3px;
  position:relative;overflow:hidden;
}
.btn::after{
  content:'';position:absolute;inset:0;
  background:white;opacity:0;transition:opacity .2s;
}
.btn:hover::after{opacity:.07;}
.btn:active{transform:scale(.97);}
.btn:disabled{opacity:.35;cursor:not-allowed;transform:none!important;}

.btn-p{background:var(--accent1);color:#080a0f;}
.btn-p:hover{box-shadow:0 0 20px var(--glow1);transform:translateY(-1px);}
.btn-s{background:transparent;color:var(--accent3);border:1px solid rgba(77,184,255,.4);}
.btn-s:hover{background:var(--glow3);box-shadow:0 0 16px var(--glow3);transform:translateY(-1px);}
.btn-g{background:transparent;color:var(--muted2);border:1px solid var(--border2);}
.btn-g:hover{color:var(--text);border-color:var(--muted2);}

.spinner{
  display:none;width:14px;height:14px;
  border:2px solid rgba(255,255,255,.2);
  border-top-color:currentColor;
  border-radius:50%;
  animation:spin .6s linear infinite;
}
@keyframes spin{to{transform:rotate(360deg)}}
.loading .spinner{display:block;}
.loading .blabel{display:none;}
.loading{pointer-events:none;}

/* ── Chips ── */
.chips-row{
  display:flex;flex-wrap:wrap;gap:8px;
  margin-bottom:16px;
}
.info-chip{
  font-family:var(--font-mono);font-size:10px;
  padding:5px 12px;border-radius:6px;
  background:var(--surface2);border:1px solid var(--border2);
  color:var(--muted2);
}
.info-chip b{font-weight:600;}
.info-chip.green b{color:var(--accent1);}
.info-chip.blue  b{color:var(--accent3);}
.info-chip.pink  b{color:var(--accent2);}
.info-chip.gold  b{color:var(--accent4);}

/* ── Tabs ── */
.tabs{
  display:flex;
  border-bottom:1px solid var(--border2);
  margin-bottom:24px;gap:0;
}
.tab-btn{
  background:none;border:none;
  border-bottom:2px solid transparent;
  color:var(--muted2);
  font-family:var(--font-sans);font-size:12px;font-weight:700;
  letter-spacing:.5px;
  padding:10px 20px;cursor:pointer;
  transition:all .2s;margin-bottom:-1px;
}
.tab-btn.active{color:var(--accent1);border-bottom-color:var(--accent1);}
.tab-panel{display:none;}
.tab-panel.active{display:block;animation:fadeIn .3s ease;}
@keyframes fadeIn{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:none}}

/* ── Prediction chips ── */
.pred-grid{
  display:flex;flex-wrap:wrap;gap:10px;
  min-height:60px;margin-bottom:24px;
}
.pred-chip{
  display:flex;flex-direction:column;align-items:center;gap:5px;
  padding:12px 20px;
  background:var(--surface2);
  border:1px solid var(--border2);
  border-radius:10px;cursor:pointer;
  transition:all .22s;
  animation:chipIn .28s ease both;
  min-width:100px;
}
.pred-chip:hover{transform:translateY(-3px);}
.pred-chip .pw{font-family:var(--font-mono);font-size:16px;font-weight:600;}
.pred-chip .pp{font-family:var(--font-mono);font-size:10px;color:var(--muted2);}
@keyframes chipIn{
  from{opacity:0;transform:translateY(8px) scale(.93)}
  to  {opacity:1;transform:none}
}

/* ── Bar chart ── */
.bar-list{display:flex;flex-direction:column;gap:8px;}
.bar-row{display:flex;align-items:center;gap:12px;}
.bar-lbl{
  font-family:var(--font-mono);font-size:12px;
  width:120px;flex-shrink:0;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
}
.bar-track{
  flex:1;height:6px;
  background:var(--border2);border-radius:99px;overflow:hidden;
}
.bar-fill{height:100%;border-radius:99px;transition:width .55s cubic-bezier(.4,0,.2,1);}
.bar-val{
  font-family:var(--font-mono);font-size:10px;
  color:var(--muted2);width:60px;text-align:right;
}

/* ── Completion box ── */
.completion-box{
  background:var(--surface2);
  border:1px solid var(--border2);
  border-radius:12px;
  padding:20px 22px;
  font-family:var(--font-mono);font-size:16px;
  line-height:1.8;min-height:70px;
  word-break:break-word;
}
.seed-w{color:var(--text);}
.gen-w{color:var(--accent2);}
.cursor-blink{
  display:inline-block;width:2px;height:1.1em;
  background:var(--accent1);
  margin-left:3px;vertical-align:middle;
  animation:blink 1s step-end infinite;
}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0}}

.comp-actions{display:flex;gap:10px;margin-top:12px;}

/* ── History ── */
.hist-list{display:flex;flex-direction:column;gap:8px;}
.hist-item{
  background:var(--surface2);
  border:1px solid var(--border2);
  border-radius:10px;
  padding:12px 16px;
  cursor:pointer;transition:border-color .2s;
  display:flex;justify-content:space-between;
  align-items:flex-start;gap:12px;
}
.hist-item:hover{border-color:var(--accent3);}
.hist-meta{
  font-family:var(--font-mono);font-size:9px;
  letter-spacing:2px;text-transform:uppercase;
  color:var(--muted);margin-bottom:5px;
}
.hist-text{font-family:var(--font-mono);font-size:13px;}
.hist-gen{color:var(--accent2);}
.hist-del{
  background:none;border:none;color:var(--muted);
  cursor:pointer;font-size:14px;
  padding:2px 6px;border-radius:4px;
  transition:all .2s;flex-shrink:0;
}
.hist-del:hover{color:var(--accent2);background:rgba(255,110,180,.1);}

/* ── Empty ── */
.empty{
  text-align:center;padding:28px 0;
  font-family:var(--font-mono);font-size:12px;
  color:var(--muted);letter-spacing:1px;
}

/* ── Toast ── */
#toast{
  position:fixed;bottom:28px;right:28px;
  background:var(--accent1);color:#080a0f;
  padding:10px 22px;border-radius:8px;
  font-family:var(--font-mono);font-size:12px;font-weight:600;
  opacity:0;transform:translateY(12px);
  transition:all .3s;pointer-events:none;z-index:999;
  box-shadow:0 0 24px var(--glow1);
}
#toast.show{opacity:1;transform:none;}

/* ── Loading overlay ── */
#loading-overlay{
  position:fixed;inset:0;z-index:99;
  background:var(--bg);
  display:flex;flex-direction:column;
  align-items:center;justify-content:center;gap:20px;
  transition:opacity .5s;
}
#loading-overlay.hidden{opacity:0;pointer-events:none;}
.loader-ring{
  width:60px;height:60px;
  border:3px solid var(--border2);
  border-top-color:var(--accent1);
  border-radius:50%;
  animation:spin .8s linear infinite;
}
.loader-text{
  font-family:var(--font-mono);font-size:12px;
  letter-spacing:3px;text-transform:uppercase;
  color:var(--muted2);
}

@media(max-width:600px){
  .wrap{padding:24px 14px 60px;}
  .stats-bar{display:none;}
  .bar-lbl{width:80px;font-size:10px;}
  .card{padding:18px;}
}
</style>
</head>
<body>

<!-- Loading overlay -->
<div id="loading-overlay">
  <div class="loader-ring"></div>
  <div class="loader-text" id="loader-msg">Loading NLP models…</div>
</div>

<div class="wrap">

  <!-- Header -->
  <div class="header">
    <div class="header-eyebrow"><div class="dot"></div>NUTECH · AI-23 · NLP Lab</div>
    <h1>Sentence Completion<br>&amp; Next Word Predictor</h1>
    <p class="sub">Bigram &amp; Trigram Language Models · WikiText-2 Dataset · Laplace Smoothing</p>
    <div class="tags">
      <span class="tag tag-green">N-gram</span>
      <span class="tag tag-blue">WikiText-2</span>
      <span class="tag tag-pink">Laplace Smoothing</span>
      <span class="tag tag-gold">UNK Handling</span>
    </div>
  </div>

  <!-- Stats bar -->
  <div class="stats-bar">
    <div class="stat"><span class="stat-val" id="sv">—</span><span class="stat-lbl">Vocab Size</span></div>
    <div class="stat"><span class="stat-val" id="sb">—</span><span class="stat-lbl">Bigram Contexts</span></div>
    <div class="stat"><span class="stat-val" id="st">—</span><span class="stat-lbl">Trigram Contexts</span></div>
    <div class="stat"><span class="stat-val">WikiText-2</span><span class="stat-lbl">Dataset</span></div>
  </div>

  <!-- Input card -->
  <div class="card">
    <div class="slabel">Input</div>

    <!-- Controls -->
    <div class="controls">
      <div>
        <div class="seg" id="modelSeg">
          <button class="seg-btn active" onclick="setModel('bigram')">Bigram</button>
          <button class="seg-btn"        onclick="setModel('trigram')">Trigram</button>
        </div>
      </div>
      <div class="range-wrap">
        <span>Max words:</span>
        <input type="range" id="maxWords" min="3" max="20" value="10"
               oninput="document.getElementById('rv').textContent=this.value">
        <span class="range-val" id="rv">10</span>
      </div>
    </div>

    <!-- Textarea -->
    <div class="input-wrap">
      <textarea id="inp"
                placeholder="Start typing… e.g. 'the scientists discovered that'"
                oninput="onType()"></textarea>
      <div class="input-footer">
        <span id="wc">0 words</span>
        <span id="cc">0 chars</span>
      </div>
    </div>

    <!-- Actions -->
    <div class="action-row">
      <button class="btn btn-p" id="bPredict" onclick="doPredict()" disabled>
        <div class="spinner"></div><span class="blabel">⚡ Predict Next Word</span>
      </button>
      <button class="btn btn-s" id="bComplete" onclick="doComplete()" disabled>
        <div class="spinner"></div><span class="blabel">✨ Complete Sentence</span>
      </button>
      <button class="btn btn-g" onclick="doClear()">✕ Clear</button>
    </div>

    <!-- Info chips -->
    <div class="chips-row" style="margin-top:16px;margin-bottom:0">
      <div class="info-chip green">Model: <b id="modelChip">Bigram</b></div>
      <div class="info-chip blue">Dataset: <b>WikiText-2</b></div>
      <div class="info-chip pink">Smoothing: <b>Laplace</b></div>
      <div class="info-chip gold">Backoff: <b>Bigram fallback</b></div>
    </div>
  </div>

  <!-- Results card -->
  <div class="card">
    <div class="tabs">
      <button class="tab-btn active" onclick="tab('predictions',this)">Next Word</button>
      <button class="tab-btn"        onclick="tab('completion',this)">Completion</button>
      <button class="tab-btn"        onclick="tab('history',this)">History</button>
    </div>

    <!-- Predictions -->
    <div class="tab-panel active" id="tab-predictions">
      <div class="slabel">Top 5 Candidates</div>
      <div class="pred-grid" id="predGrid">
        <div class="empty">Run a prediction to see results ↑</div>
      </div>
      <div class="slabel">Probability Distribution</div>
      <div class="bar-list" id="barList"></div>
    </div>

    <!-- Completion -->
    <div class="tab-panel" id="tab-completion">
      <div class="slabel">Generated Output</div>
      <div class="completion-box" id="compBox">
        <span style="color:var(--muted);font-size:13px">
          Click "Complete Sentence" to generate continuation…
        </span>
      </div>
      <div class="comp-actions">
        <button class="btn btn-g" onclick="copyComp()">⎘ Copy</button>
        <button class="btn btn-g" onclick="useComp()">↑ Use as input</button>
      </div>
    </div>

    <!-- History -->
    <div class="tab-panel" id="tab-history">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px">
        <div class="slabel" style="margin:0">Session History</div>
        <button class="btn btn-g" style="padding:6px 12px;font-size:11px" onclick="clearHist()">Clear all</button>
      </div>
      <div class="hist-list" id="histList"><div class="empty">No history yet</div></div>
    </div>
  </div>

</div><!-- .wrap -->

<div id="toast"></div>

<script>
// ── State ──────────────────────────────────────────────────────────────────
let model = 'bigram';
let lastComp = {full:'', added:[], seed:''};
let hist = [];
const COLORS = ['#00ffb3','#4db8ff','#ffd166','#ff6eb4','#c084fc'];

// ── Boot: load health stats ────────────────────────────────────────────────
(async function boot(){
  try{
    const r = await fetch('/health');
    const d = await r.json();
    document.getElementById('sv').textContent = fmt(d.vocab_size);
    document.getElementById('sb').textContent = fmt(d.bigram_contexts);
    document.getElementById('st').textContent = fmt(d.trigram_contexts);
  }catch(e){}
  const ol = document.getElementById('loading-overlay');
  ol.classList.add('hidden');
  setTimeout(()=>ol.remove(), 600);
})();

function fmt(n){
  if(n>=1e6) return (n/1e6).toFixed(1)+'M';
  if(n>=1e3) return (n/1e3).toFixed(1)+'K';
  return n;
}

// ── Helpers ────────────────────────────────────────────────────────────────
function setModel(m){
  model = m;
  document.querySelectorAll('.seg-btn').forEach((b,i)=>{
    b.classList.toggle('active', (i===0&&m==='bigram')||(i===1&&m==='trigram'));
  });
  document.getElementById('modelChip').textContent =
    m==='bigram' ? 'Bigram' : 'Trigram';
}

function onType(){
  const v = document.getElementById('inp').value;
  const words = v.trim() ? v.trim().split(/\s+/).length : 0;
  document.getElementById('wc').textContent = words+' words';
  document.getElementById('cc').textContent = v.length+' chars';
  const has = v.trim().length > 0;
  document.getElementById('bPredict').disabled  = !has;
  document.getElementById('bComplete').disabled = !has;
}

function setLoad(id, on){
  const b = document.getElementById(id);
  b.classList.toggle('loading', on);
  b.disabled = on;
}

function toast(msg){
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  clearTimeout(t._t);
  t._t = setTimeout(()=>t.classList.remove('show'), 2400);
}

function tab(name, btn){
  document.querySelectorAll('.tab-panel').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
  document.getElementById('tab-'+name).classList.add('active');
  btn.classList.add('active');
}

// ── Predict ────────────────────────────────────────────────────────────────
async function doPredict(){
  const text = document.getElementById('inp').value.trim();
  if(!text) return;
  setLoad('bPredict', true);
  try{
    const r = await fetch('/predict',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({text, mode: model})
    });
    const preds = await r.json();
    renderPreds(preds);
    tab('predictions', document.querySelectorAll('.tab-btn')[0]);
  }catch(e){ toast('Error — server may be loading'); }
  finally{ setLoad('bPredict', false); document.getElementById('bPredict').disabled=false; }
}

function renderPreds(preds){
  const grid = document.getElementById('predGrid');
  const bars = document.getElementById('barList');
  grid.innerHTML = ''; bars.innerHTML = '';
  if(!preds.length){
    grid.innerHTML = '<div class="empty">No predictions returned</div>';
    return;
  }
  const max = preds[0].prob;
  preds.forEach((p,i)=>{
    // chip
    const chip = document.createElement('div');
    chip.className = 'pred-chip';
    chip.style.cssText = `border-color:${COLORS[i]}33;animation-delay:${i*55}ms`;
    chip.innerHTML = `<span class="pw" style="color:${COLORS[i]}">${p.word}</span>`+
                     `<span class="pp">${p.prob.toFixed(4)}%</span>`;
    chip.onclick = ()=>appendWord(p.word);
    grid.appendChild(chip);
    // bar
    const row = document.createElement('div');
    row.className = 'bar-row';
    row.innerHTML = `<div class="bar-lbl" style="color:${COLORS[i]}">${p.word}</div>`+
                    `<div class="bar-track"><div class="bar-fill" style="width:0%;background:${COLORS[i]}"></div></div>`+
                    `<div class="bar-val">${p.prob.toFixed(4)}%</div>`;
    bars.appendChild(row);
    requestAnimationFrame(()=>setTimeout(()=>{
      row.querySelector('.bar-fill').style.width = (max>0?(p.prob/max)*100:0)+'%';
    }, 60+i*90));
  });
}

function appendWord(w){
  const ta = document.getElementById('inp');
  ta.value = (ta.value.trimEnd()+' '+w).trimStart();
  onType(); toast('Appended: "'+w+'"');
}

// ── Complete ───────────────────────────────────────────────────────────────
async function doComplete(){
  const text = document.getElementById('inp').value.trim();
  if(!text) return;
  const mw = document.getElementById('maxWords').value;
  setLoad('bComplete', true);
  // show cursor animation
  document.getElementById('compBox').innerHTML =
    '<span class="seed-w">'+text+'</span> <span class="cursor-blink"></span>';
  tab('completion', document.querySelectorAll('.tab-btn')[1]);
  try{
    const r = await fetch('/complete',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({text, mode: model, max_words: mw})
    });
    const result = await r.json();
    lastComp = {...result, seed: text};
    renderComp(text, result);
    addHist(text, result);
  }catch(e){ toast('Error — server may be loading'); }
  finally{ setLoad('bComplete', false); document.getElementById('bComplete').disabled=false; }
}

function renderComp(seed, result){
  const box = document.getElementById('compBox');
  const added = result.added||[];
  box.innerHTML = '<span class="seed-w">'+seed+'</span>'+
    (added.length ? ' <span class="gen-w">'+added.join(' ')+'</span>' :
     '<span style="color:var(--muted)"> (end of sequence)</span>');
}

function copyComp(){
  const txt = document.getElementById('compBox').innerText;
  navigator.clipboard.writeText(txt)
    .then(()=>toast('Copied!'))
    .catch(()=>toast('Copy failed'));
}

function useComp(){
  if(!lastComp.full) return;
  document.getElementById('inp').value = lastComp.full;
  onType();
  tab('predictions', document.querySelectorAll('.tab-btn')[0]);
  toast('Loaded into input');
}

// ── History ────────────────────────────────────────────────────────────────
function addHist(seed, result){
  hist.unshift({seed, result, time: new Date().toLocaleTimeString()});
  if(hist.length>30) hist.pop();
  renderHist();
}

function renderHist(){
  const list = document.getElementById('histList');
  if(!hist.length){ list.innerHTML='<div class="empty">No history yet</div>'; return; }
  list.innerHTML = hist.map((h,i)=>`
    <div class="hist-item" onclick="loadHist(${i})">
      <div style="flex:1;min-width:0">
        <div class="hist-meta">${h.time}</div>
        <div class="hist-text">
          ${h.seed}
          ${h.result.added&&h.result.added.length
            ? ' <span class="hist-gen">'+h.result.added.join(' ')+'</span>' : ''}
        </div>
      </div>
      <button class="hist-del" onclick="event.stopPropagation();delHist(${i})">✕</button>
    </div>`).join('');
}

function loadHist(i){
  const h = hist[i];
  document.getElementById('inp').value = h.result.full||h.seed;
  onType(); renderComp(h.seed, h.result);
  tab('completion', document.querySelectorAll('.tab-btn')[1]);
}
function delHist(i){ hist.splice(i,1); renderHist(); }
function clearHist(){ hist=[]; renderHist(); }

// ── Clear ──────────────────────────────────────────────────────────────────
function doClear(){
  document.getElementById('inp').value='';
  onType();
  document.getElementById('predGrid').innerHTML='<div class="empty">Run a prediction to see results ↑</div>';
  document.getElementById('barList').innerHTML='';
  document.getElementById('compBox').innerHTML='<span style="color:var(--muted);font-size:13px">Click "Complete Sentence" to generate continuation…</span>';
}

// ── Keyboard shortcut ──────────────────────────────────────────────────────
document.addEventListener('keydown', e=>{
  if((e.ctrlKey||e.metaKey)&&e.key==='Enter'){
    e.preventDefault();
    if(!document.getElementById('bPredict').disabled) doPredict();
  }
  if((e.ctrlKey||e.metaKey)&&e.shiftKey&&e.key==='Enter'){
    e.preventDefault();
    if(!document.getElementById('bComplete').disabled) doComplete();
  }
});
</script>
</body>
</html>"""

# ── 7. Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
