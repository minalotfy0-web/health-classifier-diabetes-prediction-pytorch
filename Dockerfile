# Web demo image: docker build -t diabetes-demo . && docker run -p 7860:7860 diabetes-demo
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    PORT=7860

WORKDIR /app

# CPU-only torch first, so the pinned requirements do not pull the CUDA build.
RUN pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install -r requirements.txt ".[web]"

COPY web ./web
COPY results/tuned/model.joblib ./results/tuned/model.joblib

RUN useradd --create-home --uid 1000 app
USER app

EXPOSE 7860
CMD ["sh", "-c", "uvicorn web.app:app --host 0.0.0.0 --port ${PORT}"]
