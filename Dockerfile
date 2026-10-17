FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY app ./app

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir .

EXPOSE 8787

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8787"]
