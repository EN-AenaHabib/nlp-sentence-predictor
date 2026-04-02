# 🧠 NLP Sentence Completion & Next Word Predictor

<div align="center">

[![Live Demo](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-yellow?style=for-the-badge)](https://huggingface.co/spaces/YOUR_USERNAME/nlp-sentence-predictor)
[![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-black?style=for-the-badge&logo=flask)](https://flask.palletsprojects.com)
[![Dataset](https://img.shields.io/badge/Dataset-WikiText--2-green?style=for-the-badge)](https://huggingface.co/datasets/wikitext)
[![License](https://img.shields.io/badge/License-MIT-red?style=for-the-badge)](LICENSE)

**National University of Technology · Department of Computer Science**  
**Course:** Natural Language Processing Lab | **Batch:** AI-23

</div>

---

## 🔴 Live Demo

> 👉 **[Click here to try the live demo](https://huggingface.co/spaces/YOUR_USERNAME/nlp-sentence-predictor)**

---

## ✨ Features

- **Bigram & Trigram** language models with Laplace (Add-1) smoothing
- **WikiText-2** dataset — high-quality English Wikipedia text (~2M tokens)
- **UNK handling** — rare words replaced with `<UNK>` token
- **Trigram → Bigram backoff** when context is unseen
- **Top-5 next word predictions** with animated probability bars
- **Greedy sentence completion** with highlighted generated words
- **Session history** — click any past result to reload it
- **Keyboard shortcuts** — `Ctrl+Enter` predict, `Ctrl+Shift+Enter` complete
- **Fully custom HTML/CSS/JS frontend** — dark futuristic design

---

## 🗂️ Project Structure

```
nlp-predictor/
├── app.py            ← Flask backend + NLP models + HTML frontend
├── requirements.txt  ← Python dependencies
├── Dockerfile        ← For Hugging Face Spaces (Docker SDK)
└── README.md         ← This file
```

---

## 🚀 Deploy to Hugging Face Spaces

### Step 1 — Create a Space
1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click **"Create new Space"**
3. Name it `nlp-sentence-predictor`
4. Select **Docker** as the SDK
5. Set visibility to **Public**

### Step 2 — Push your code
```bash
# Clone your new space repo
git clone https://huggingface.co/spaces/YOUR_USERNAME/nlp-sentence-predictor
cd nlp-sentence-predictor

# Copy the project files in
cp /path/to/app.py .
cp /path/to/requirements.txt .
cp /path/to/Dockerfile .
cp /path/to/README.md .

# Push
git add .
git commit -m "Initial deployment"
git push
```

### Step 3 — Wait ~3 minutes
Hugging Face will build the Docker image and start your app automatically.  
Your live URL will be: `https://YOUR_USERNAME-nlp-sentence-predictor.hf.space`

---

## 💻 Run Locally

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/nlp-sentence-predictor
cd nlp-sentence-predictor

# Install dependencies
pip install -r requirements.txt

# Run
python app.py
```
Open `http://localhost:7860` in your browser.

---

## 🧪 How It Works

### Dataset
**WikiText-2** is a collection of high-quality Wikipedia articles with ~2 million tokens. It's much richer than the Brown Corpus for sentence completion because:
- Longer, more coherent sentences
- Diverse topics (science, history, geography, etc.)
- Clean punctuation and formatting

### N-gram Models

**Bigram model** — predicts next word from 1 previous word:
```
P(w_i | w_{i-1}) = count(w_{i-1}, w_i) + 1
                   ─────────────────────────
                   count(w_{i-1}) + |V|
```

**Trigram model** — predicts next word from 2 previous words:
```
P(w_i | w_{i-2}, w_{i-1}) = count(w_{i-2}, w_{i-1}, w_i) + 1
                             ──────────────────────────────────
                             count(w_{i-2}, w_{i-1}) + |V|
```

**Laplace (Add-1) Smoothing** ensures unseen word combinations get a small non-zero probability.

**Backoff**: When the trigram context has zero observations, the model falls back to the bigram prediction automatically.

---

## 📊 Model Stats

| Metric | Value |
|--------|-------|
| Dataset | WikiText-2 (~2M tokens) |
| Vocabulary | ~10,000 words (min_freq=3) |
| Training split | 90% |
| Smoothing | Laplace (Add-1) |
| Backoff | Trigram → Bigram |

---

## 📚 Lab Context

This project was built as part of:
- **Lab 5**: Spelling corrector using N-grams
- **Lab 6**: Text Classification & Spam Detection (BoW + Naive Bayes)
- **This project**: Sentence Completion & Next Word Prediction (deployed)

---

## 👩‍💻 Author

**Aena Habib** — F23607020  
Submitted to: LE Sabahat Fatima  
National University of Technology (NUTECH)

---

## 📄 License

MIT License — free to use and modify.
