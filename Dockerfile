FROM python:3.12-slim

# RUN apt-get update && apt-get upgrade -y && \
#     apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# install deps first so this layer caches across rebuilds
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN useradd -m appuser

COPY --chown=appuser:appuser . .
RUN mkdir -p /app/logs && chown appuser:appuser /app/logs

EXPOSE 8000

USER appuser

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
