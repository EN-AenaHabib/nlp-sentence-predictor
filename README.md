

```markdown
# <div align="center" style="color:#4DA6FF;">NLP Sentence Predictor</div>

<div align="center">

[![Live Demo](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-yellow?style=for-the-badge)](https://huggingface.co/spaces/Aenpi/nlp-sentence-predictor)  
[![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)](https://python.org)  
[![Flask](https://img.shields.io/badge/Flask-3.0-black?style=for-the-badge&logo=flask)](https://flask.palletsprojects.com)  
[![Dataset](https://img.shields.io/badge/Dataset-WikiText--2-green?style=for-the-badge)](https://huggingface.co/datasets/wikitext)  
[![License](https://img.shields.io/badge/License-MIT-red?style=for-the-badge)](LICENSE)

**National University of Technology · Department of Computer Science**  
**Course:** Natural Language Processing Lab | **Batch:** AI-23

</div>

---

## 🔴 Live Demo

> 👉 **[Try it now!](https://huggingface.co/spaces/Aenpi/nlp-sentence-predictor)**  

Watch your sentences come alive with **next-word predictions** and **dynamic completion**!

---

## ✨ Features

- **Bigram & Trigram** language models with Laplace (Add-1) smoothing  
- **WikiText-2** dataset (~2M tokens) — coherent, high-quality Wikipedia text  
- **UNK handling** — rare words replaced with `<UNK>`  
- **Trigram → Bigram backoff** for unseen contexts  
- **Top-5 next word predictions** with animated probability bars  
- **Greedy sentence completion** with highlighted generated words  
- **Session history** — click any past result to reload it  
- **Keyboard shortcuts:** `Ctrl+Enter` to predict, `Ctrl+Shift+Enter` to complete  
- **Modern HTML/CSS/JS frontend** — dark futuristic theme  

---

## 🗂️ Project Structure

```

nlp-predictor/
├── app.py            ← Flask backend + NLP models + HTML frontend
├── requirements.txt  ← Python dependencies
├── Dockerfile        ← For Hugging Face Spaces (Docker SDK)
└── README.md         ← This file

````

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
# Clone your Space repo
git clone https://huggingface.co/spaces/Aenpi/nlp-sentence-predictor
cd nlp-sentence-predictor

# Add files if not already present
git add .
git commit -m "Deploy NLP Sentence Predictor"
git push
````

### Step 3 — Wait for build

* Hugging Face auto-builds the Docker image
* Your live URL: `https://Aenpi-nlp-sentence-predictor.hf.space`

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

**Dataset:** WikiText-2 (~2M tokens) — clean, coherent Wikipedia text

**Bigram model:**

```
P(w_i | w_{i-1}) = (count(w_{i-1}, w_i) + 1) / (count(w_{i-1}) + |V|)
```

**Trigram model:**

```
P(w_i | w_{i-2}, w_{i-1}) = (count(w_{i-2}, w_{i-1}, w_i) + 1) / (count(w_{i-2}, w_{i-1}) + |V|)
```

**Backoff:** Trigram → Bigram when context is unseen
**Laplace Smoothing:** ensures rare combinations get non-zero probability

---

## 📊 Model Stats

| Metric         | Value                      |
| -------------- | -------------------------- |
| Dataset        | WikiText-2 (~2M tokens)    |
| Vocabulary     | ~10,000 words (min_freq=3) |
| Training split | 90%                        |
| Smoothing      | Laplace (Add-1)            |
| Backoff        | Trigram → Bigram           |

---

## Author

**Aena Habib — F23607020**

---

## 📄 License

MIT License — free to use and modify.

```

---

Do you want me to do that?
```
