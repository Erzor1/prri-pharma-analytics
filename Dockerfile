FROM python:3.10-slim

LABEL maintainer="Pharma R&D Data Analyst"
LABEL description="PRRI — Мониторинг и скоринг технологического брака лекарственных средств (Росздравнадзор)"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Установка системных утилит (curl для healthcheck, libpq для psycopg2)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# При старте контейнера инициализируем/обновляем БД (PostgreSQL/SQLite) и запускаем Streamlit
CMD ["sh", "-c", "python -m src.pipeline && streamlit run app.py --server.port=8501 --server.address=0.0.0.0"]
