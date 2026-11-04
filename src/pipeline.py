"""
Единый оркестратор ETL-пайплайна проекта PRRI:
1. Выгрузка данных из API Росздравнадзора (полная или инкрементальная).
2. Очистка, нормализация, доменная классификация брака и расчет PRRI (src/cleaner.py).
3. Загрузка таблиц и аналитических SQL-витрин в PostgreSQL 16 / SQLite (src/database.py).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from src.cleaner import build_cleaned_dataset
from src.database import init_database, verify_views
from src.scraper import RoszdravnadzorScraper

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("rzn_pipeline")

RAW_CSV_PATH = Path("data/raw/rzn_full_raw.csv")
CLEANED_CSV_PATH = Path("data/processed/rzn_cleaned.csv")
MFG_CSV_PATH = Path("data/processed/manufacturers_prri.csv")


def run_pipeline(incremental: bool = False, force_scrape: bool = False) -> None:
    """Запускает полный или инкрементальный цикл обновления данных и витрин БД."""
    scraper = RoszdravnadzorScraper()

    if not RAW_CSV_PATH.exists() or force_scrape:
        logger.info("Запуск полной выгрузки реестра Росздравнадзора с 01.01.2020...")
        rows, _ = scraper.fetch_records(date_from="01.01.2020", page_size=500, max_records=None)
        RAW_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(RAW_CSV_PATH, index=False, encoding="utf-8-sig")
    elif incremental:
        logger.info("Запуск инкрементальной догрузки свежих писем Росздравнадзора...")
        fresh_rows, _ = scraper.fetch_records(date_from="01.01.2026", page_size=200, max_records=300)
        df_fresh = pd.DataFrame(fresh_rows)
        df_old = pd.read_csv(RAW_CSV_PATH)
        before_cnt = len(df_old)
        df_combined = pd.concat([df_fresh, df_old], ignore_index=True).drop_duplicates(
            subset=["letter_raw", "trade_name", "batch_number", "manufacturer_raw", "defect_parameter"]
        )
        added = max(0, len(df_combined) - before_cnt)
        df_combined.to_csv(RAW_CSV_PATH, index=False, encoding="utf-8-sig")
        logger.info("Инкрементальное обновление завершено: добавлено %d новых записей.", added)
    else:
        logger.info("Используется существующий сырой датасет: %s", RAW_CSV_PATH)

    build_cleaned_dataset(
        raw_csv_path=RAW_CSV_PATH,
        cleaned_csv_path=CLEANED_CSV_PATH,
        mfg_csv_path=MFG_CSV_PATH,
    )

    engine = init_database(
        cleaned_csv_path=CLEANED_CSV_PATH,
        mfg_csv_path=MFG_CSV_PATH,
    )
    verify_views(engine)


def main() -> None:
    parser = argparse.ArgumentParser(description="ETL-оркестратор проекта PRRI")
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Догрузить только свежие письма из API Росздравнадзора",
    )
    parser.add_argument(
        "--force-scrape",
        action="store_true",
        help="Принудительно перекачать весь архив с 01.01.2020",
    )
    args = parser.parse_args()
    run_pipeline(incremental=args.incremental, force_scrape=args.force_scrape)


if __name__ == "__main__":
    main()
