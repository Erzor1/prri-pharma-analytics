"""
Слой работы с реляционной БД (PostgreSQL 16 / SQLite fallback) на базе SQLAlchemy.

Архитектура:
- В продакшене (Docker Compose на VPS) использует PostgreSQL 16 через переменную окружения DATABASE_URL:
  postgresql+psycopg2://pharma_user:pharma_pass@postgres:5432/pharma_prri
- При локальном запуске без поднятого контейнера PostgreSQL автоматически переключается
  на локальную реляционную базу SQLite (data/processed/pharma_quality.db) с теми же таблицами и SQL-витринами.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("rzn_database")

DEFAULT_SQLITE_PATH = Path("data/processed/pharma_quality.db")
DEFAULT_VIEWS_SQL_PATH = Path("sql/views.sql")


def get_db_engine(db_url: str | None = None) -> Engine:
    """
    Создает SQLAlchemy Engine.
    Приоритет:
      1) Явно переданный db_url
      2) Переменная окружения DATABASE_URL (PostgreSQL в Docker/VPS)
      3) Локальный файл SQLite data/processed/pharma_quality.db
    """
    target_url = db_url or os.getenv("DATABASE_URL", "").strip()
    if target_url:
        try:
            engine = create_engine(target_url, pool_pre_ping=True)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Успешное подключение к СУБД: %s", engine.dialect.name.upper())
            return engine
        except Exception as exc:
            logger.warning(
                "Не удалось подключиться к DATABASE_URL (%s). Переключаемся на локальную SQLite. Причина: %s",
                target_url.split("@")[-1],
                exc,
            )

    DEFAULT_SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    sqlite_url = f"sqlite:///{DEFAULT_SQLITE_PATH.as_posix()}"
    engine = create_engine(sqlite_url)
    logger.info("Используется локальная СУБД SQLite: %s", DEFAULT_SQLITE_PATH)
    return engine


def _split_sql_statements(sql_script: str) -> list[str]:
    """Разбивает SQL-скрипт с несколькими командами DROP VIEW / CREATE VIEW на отдельные выражения."""
    statements: list[str] = []
    current_lines: list[str] = []

    for line in sql_script.splitlines():
        stripped = line.strip()
        if stripped.startswith("--"):
            continue
        current_lines.append(line)
        if stripped.endswith(";"):
            stmt = "\n".join(current_lines).strip()
            if stmt.endswith(";"):
                stmt = stmt[:-1].strip()
            if stmt:
                statements.append(stmt)
            current_lines = []

    remainder = "\n".join(current_lines).strip()
    if remainder:
        statements.append(remainder)
    return statements


def init_database(
    cleaned_csv_path: str | Path = "data/processed/rzn_cleaned.csv",
    mfg_csv_path: str | Path = "data/processed/manufacturers_prri.csv",
    views_sql_path: str | Path = DEFAULT_VIEWS_SQL_PATH,
    db_url: str | None = None,
) -> Engine:
    """
    Загружает очищенные таблицы (dim_manufacturers, fact_recalls) в СУБД (PostgreSQL / SQLite),
    создает индексы и разворачивает аналитические SQL-витрины из sql/views.sql.
    """
    engine = get_db_engine(db_url=db_url)

    logger.info("Чтение подготовленных CSV-файлов для загрузки в БД...")
    df_recalls = pd.read_csv(cleaned_csv_path)
    df_mfg = pd.read_csv(mfg_csv_path)

    # Приводим булевы флаги к целочисленному виду (0/1) для 100% идентичной работы
    # условий WHERE is_recall_event = 1 и SUM(CASE WHEN has_explicit_defect = 1 ...)
    # как в PostgreSQL 16, так и в SQLite 3.37+
    for bool_col in ("has_explicit_defect", "is_tech_or_critical_defect", "is_recall_event"):
        if bool_col in df_recalls.columns:
            df_recalls[bool_col] = df_recalls[bool_col].astype(bool).astype(int)

    with engine.begin() as conn:
        # Перед перезаливкой таблиц в PostgreSQL/SQLite удаляем зависимые витрины
        for view_name in (
            "v_monthly_defect_dynamics",
            "v_manufacturer_risk_ranking",
            "v_dosage_form_defect_matrix",
            "v_chronic_drug_recidivism",
        ):
            conn.execute(text(f"DROP VIEW IF EXISTS {view_name}"))

        logger.info("Загрузка таблицы dim_manufacturers (%d строк)...", len(df_mfg))
        df_mfg.to_sql("dim_manufacturers", con=conn, if_exists="replace", index=False)

        logger.info("Загрузка таблицы fact_recalls (%d строк)...", len(df_recalls))
        df_recalls.to_sql("fact_recalls", con=conn, if_exists="replace", index=False, chunksize=2000)

        # Создаем индексы для ускорения аналитических запросов и витрин
        index_statements = [
            "CREATE INDEX IF NOT EXISTS idx_fact_recalls_date ON fact_recalls (letter_date)",
            "CREATE INDEX IF NOT EXISTS idx_fact_recalls_ym ON fact_recalls (year_month)",
            "CREATE INDEX IF NOT EXISTS idx_fact_recalls_mfg ON fact_recalls (manufacturer_clean)",
            "CREATE INDEX IF NOT EXISTS idx_fact_recalls_form ON fact_recalls (dosage_form_group)",
            "CREATE INDEX IF NOT EXISTS idx_fact_recalls_cat ON fact_recalls (defect_category)",
        ]
        for idx_sql in index_statements:
            conn.execute(text(idx_sql))

        # Выполняем скрипт создания аналитических витрин sql/views.sql
        sql_path = Path(views_sql_path)
        if sql_path.exists():
            sql_text = sql_path.read_text(encoding="utf-8")
            statements = _split_sql_statements(sql_text)
            for stmt in statements:
                conn.execute(text(stmt))
            logger.info("Успешно развернуто %d SQL-команд из %s", len(statements), sql_path)
        else:
            logger.warning("Файл SQL-витрин не найден: %s", sql_path)

    return engine


def query_df(sql_query: str, engine: Engine | None = None, params: dict | None = None) -> pd.DataFrame:
    """Выполняет SQL-запрос к БД и возвращает результат в виде pandas.DataFrame."""
    eng = engine or get_db_engine()
    with eng.connect() as conn:
        return pd.read_sql_query(text(sql_query), con=conn, params=params)


def verify_views(engine: Engine) -> None:
    """Тестирует все 4 аналитические SQL-витрины и выводит сводку в консоль."""
    views = [
        "v_monthly_defect_dynamics",
        "v_manufacturer_risk_ranking",
        "v_dosage_form_defect_matrix",
        "v_chronic_drug_recidivism",
    ]
    for view in views:
        df_view = query_df(f"SELECT * FROM {view} LIMIT 5", engine=engine)
        cnt_df = query_df(f"SELECT COUNT(*) AS cnt FROM {view}", engine=engine)
        total_rows = int(cnt_df["cnt"].iloc[0])
        logger.info("Витрина %-32s | Строк: %4d | Колонок: %2d", view, total_rows, len(df_view.columns))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Инициализация БД (PostgreSQL / SQLite) и создание аналитических SQL-витрин"
    )
    parser.add_argument(
        "--db-url",
        type=str,
        default=None,
        help="SQLAlchemy URL подключения (например postgresql+psycopg2://user:pass@localhost:5432/db)",
    )
    args = parser.parse_args()

    engine = init_database(db_url=args.db_url)
    verify_views(engine)


if __name__ == "__main__":
    main()
