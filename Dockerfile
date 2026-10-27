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

# Pre-download the exact revisions used by the runtime and evaluation manifest.
RUN python -c "\
from sentence_transformers import CrossEncoder, SentenceTransformer; \
from transformers import AutoTokenizer, AutoModelForQuestionAnswering; \
from vistaar.config import Settings; \
s=Settings(api_key=''); \
SentenceTransformer(s.embedding_model, revision=s.embedding_revision, device='cpu'); \
CrossEncoder(s.reranker_model, revision=s.reranker_revision, device='cpu'); \
AutoTokenizer.from_pretrained(s.reader_model, revision=s.reader_revision); \
AutoModelForQuestionAnswering.from_pretrained(s.reader_model, revision=s.reader_revision)"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
