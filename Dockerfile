FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
    TRANSFORMERS_OFFLINE=0

WORKDIR /app

# System deps kept minimal; torch wheels are self-contained.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY vistaar/ ./vistaar/
COPY app.py .

# Pre-download the embedding and reader models so the first request is fast.
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu'); \
from transformers import AutoTokenizer, AutoModelForQuestionAnswering; \
AutoTokenizer.from_pretrained('deepset/roberta-base-squad2'); \
AutoModelForQuestionAnswering.from_pretrained('deepset/roberta-base-squad2')"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
