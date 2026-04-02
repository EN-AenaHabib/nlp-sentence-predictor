<p align="center">
  <img src="https://img.shields.io/badge/LexPredict-NLP%20Language%20Model-4DA6FF?style=for-the-badge" alt="LexPredict"/>
</p>

<h1 align="center">LexPredict — Statistical Language Model & Sentence Predictor</h1>

<p align="center">
  Real-time next-word prediction , spell correction and sentence completion powered by N-gram language models trained on WikiText-2.
</p>

<p align="center">
  <a href="https://huggingface.co/spaces/Aenpi/nlp-sentence-predictor">
    <img src="https://img.shields.io/badge/🤗%20Live%20Demo-Hugging%20Face-yellow?style=for-the-badge" alt="Live Demo"/>
  </a>
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python" alt="Python"/>
  <img src="https://img.shields.io/badge/Flask-3.0-black?style=for-the-badge&logo=flask" alt="Flask"/>
  <img src="https://img.shields.io/badge/Dataset-WikiText--2-green?style=for-the-badge" alt="Dataset"/>
  <img src="https://img.shields.io/badge/License-MIT-red?style=for-the-badge" alt="License"/>
</p>

[![Lex-predictLive Demo](livedemo.png)](https://huggingface.co/spaces/Aenpi/nlp-sentence-predictor/embed)

---

## 🚀 Live Demo

> 👉 **[Try LexPredict Live](https://huggingface.co/spaces/Aenpi/nlp-sentence-predictor/embed)**

---

## ✨ Features

- **Bigram & Trigram language models** with Laplace (Add-1) smoothing
- **WikiText-2 dataset** (~2M tokens) — high-quality, coherent Wikipedia text
- **UNK token handling** — rare words replaced with `<UNK>` for robust coverage
- **Trigram → Bigram backoff** for graceful handling of unseen contexts
- **Top-5 next-word predictions** with animated probability bars
- **Greedy sentence completion** with highlighted generated tokens
- **Session history** — reload and compare previous prediction results
- **Keyboard shortcuts** — `Ctrl+Enter` to predict, `Ctrl+Shift+Enter` to complete
- **Responsive dark UI** — modern HTML/CSS/JS frontend

---

## 🧠 How It Works

**Dataset:** WikiText-2 (~2M tokens) sourced from high-quality Wikipedia articles.

**Bigram Model:**

$$P(w_i \mid w_{i-1}) = \frac{\text{count}(w_{i-1},\ w_i) + 1}{\text{count}(w_{i-1}) + |V|}$$

**Trigram Model:**

$$P(w_i \mid w_{i-2}, w_{i-1}) = \frac{\text{count}(w_{i-2},\ w_{i-1},\ w_i) + 1}{\text{count}(w_{i-2},\ w_{i-1}) + |V|}$$

**Backoff Strategy:** Falls back from Trigram to Bigram when the trigram context is unseen.  
**Smoothing:** Laplace (Add-1) smoothing assigns nonzero probability to all sequences.

---

## 📊 Model Statistics

| Metric         | Value                        |
|----------------|------------------------------|
| Dataset        | WikiText-2 (~2M tokens)      |
| Vocabulary     | ~10,000 words (`min_freq=3`) |
| Training Split | 90%                          |
| Smoothing      | Laplace (Add-1)              |
| Backoff        | Trigram → Bigram             |

---

## 🗂️ Project Structure
```
LexPredict/
├── app.py            # Flask backend, N-gram models, and frontend template
├── requirements.txt  # Python dependencies
├── Dockerfile        # Hugging Face Spaces (Docker SDK)
└── README.md         # Project documentation
```

---

## 💻 Run Locally
```bash
git clone https://github.com/YOUR_USERNAME/LexPredict
cd LexPredict
pip install -r requirements.txt
python app.py
```

Open `http://localhost:7860` in your browser.

---

## ☁️ Deploy to Hugging Face Spaces

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) and click **"Create new Space"**
2. Name it `lexpredict`, select **Docker** as the SDK, set visibility to **Public**
```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/lexpredict
cd lexpredict

# Copy project files in, then:
git add .
git commit -m "Deploy LexPredict"
git push
```

Hugging Face will build and host your app automatically within a few minutes.

---

## 👥 Contributors

| Name             | Email                         |
|------------------|-------------------------------|
| Aena Habib       | aenahabibf23@nutech.edu.pk    |
| Aleena Tahir     | aleenatahirf23@nutech.edu.pk  |
| Saqlain Abbas    | saqlainabbasf23@nutech.edu.pk |
| Eman Asghar Kiani| emankainif23@nutech.edu.pk    |
| Dua Kamal        | duakamalf23@nutech.edu.pk     |

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) — free to use, modify, and distribute.
