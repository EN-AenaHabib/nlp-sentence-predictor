FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY app.py .

# Hugging Face Spaces runs on port 7860
EXPOSE 7860

# Pre-download NLTK data (fallback)
RUN python -c "import nltk; nltk.download('brown', quiet=True); nltk.download('punkt', quiet=True)"

CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--timeout", "300", "--workers", "1", "app:app"]
