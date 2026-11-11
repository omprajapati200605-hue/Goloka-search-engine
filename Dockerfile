FROM python:3.12-slim

# Non-root user (Hugging Face Spaces requires uid 1000; harmless elsewhere)
RUN useradd -m -u 1000 goloka
ENV HOME=/home/goloka PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/goloka/.cache/huggingface
WORKDIR /home/goloka/app

# CPU-only PyTorch keeps the image far smaller than the default CUDA build
COPY requirements-prod.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu torch \
 && pip install -r requirements-prod.txt

COPY --chown=goloka:goloka app ./app
COPY --chown=goloka:goloka scripts ./scripts

# Download the AI model at build time so the first search is not slow
USER goloka
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Data lives here. Mount a volume on /data to keep it across restarts.
ENV GOLOKA_DATA_DIR=/data GOLOKA_DB_PATH=/data/goloka.db GOLOKA_VECTOR_DIR=/data/vectors \
    GOLOKA_API_HOST=0.0.0.0 GOLOKA_API_PORT=7860 GOLOKA_PUBLIC_READ=true
USER root
RUN mkdir -p /data && chown goloka:goloka /data
USER goloka
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request,os;urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('GOLOKA_API_PORT','7860')+'/api/v1/health')"
CMD ["python", "-m", "scripts.run_api"]
