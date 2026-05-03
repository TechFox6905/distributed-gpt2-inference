FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app

ENV HF_HOME=/app/cache
ENV TRANSFORMERS_CACHE=/app/cache
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml .
COPY src ./src

RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "gpt.api.main:app", "--host", "0.0.0.0", "--port", "8000"]