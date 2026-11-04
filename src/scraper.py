"""
Отказоустойчивый парсер API Росздравнадзора:
Сервис «Поиск изъятых из обращения лекарственных средств»
Endpoint: POST https://roszdravnadzor.gov.ru/ajax/services/lssearch
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import urllib3
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

# Обеспечиваем корректный вывод кириллицы в консоль Windows PowerShell
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("rzn_scraper")

BASE_PAGE_URL = "https://roszdravnadzor.gov.ru/services/lssearch"
AJAX_API_URL = "https://roszdravnadzor.gov.ru/ajax/services/lssearch"

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    "X-Requested-With": "XMLHttpRequest",
    "Origin": "https://roszdravnadzor.gov.ru",
    "Referer": BASE_PAGE_URL,
}

HTML_TAG_RE = re.compile(r"<[^>]+>")
HREF_RE = re.compile(r'href=["\']([^"\']+)["\']', re.IGNORECASE)
LETTER_ID_RE = re.compile(r"downloadletter=(\d+)", re.IGNORECASE)


def extract_cell_value(cell: Any, prefer_title: bool = True) -> str:
    """
    Извлекает текстовое значение из ячейки JSON ответа Росздравнадзора.
    В API поля col1..col12 обычно имеют структуру {"label": "...", "title": "..."}.
    Часто в label находится сокращенная строка, а в title — полная.
    """
    if cell is None:
        return ""
    if isinstance(cell, (str, int, float)):
        text = str(cell)
    elif isinstance(cell, dict):
        label = str(cell.get("label") or "").strip()
        title = str(cell.get("title") or "").strip()
        if prefer_title and title:
            text = title
        else:
            text = label or title
    else:
        text = str(cell)

    # Очистка от возможных HTML-тегов и декодирование HTML-сущностей (&quot;, &laquo; и т.д.)
    text = html.unescape(HTML_TAG_RE.sub(" ", text))
    return re.sub(r"\s+", " ", text).strip()


def extract_letter_meta(col10: Any) -> dict[str, str]:
    """
    Извлекает из col10 полную строку письма (номер и дату), ID письма и ссылку на скачивание PDF.
    """
    if not isinstance(col10, dict):
        raw = extract_cell_value(col10)
        return {"letter_id": "", "letter_raw": raw, "letter_link": ""}

    label_raw = str(col10.get("label") or "")
    title_raw = str(col10.get("title") or "")

    # Ищем ссылку вида ?downloadletter=... в HTML внутри label или title
    href_match = HREF_RE.search(label_raw) or HREF_RE.search(title_raw)
    letter_link = href_match.group(1) if href_match else ""
    id_match = LETTER_ID_RE.search(letter_link) or LETTER_ID_RE.search(label_raw)
    letter_id = id_match.group(1) if id_match else ""

    if letter_link and letter_link.startswith("?"):
        letter_link = f"{BASE_PAGE_URL}{letter_link}"

    clean_title = html.unescape(HTML_TAG_RE.sub(" ", title_raw)).strip()
    clean_label = html.unescape(HTML_TAG_RE.sub(" ", label_raw)).strip()

    # Выбираем наиболее информативную строку (обычно в title или label содержится '№ ... от DD.MM.YYYY')
    if clean_title and len(clean_title) >= len(clean_label):
        letter_text = clean_title
    else:
        letter_text = clean_label or clean_title

    letter_text = re.sub(r"\s+", " ", letter_text).strip()
    return {
        "letter_id": letter_id,
        "letter_raw": letter_text,
        "letter_label": re.sub(r"\s+", " ", clean_label).strip(),
        "letter_title": re.sub(r"\s+", " ", clean_title).strip(),
        "letter_link": letter_link,
    }


def parse_record(row: dict[str, Any]) -> dict[str, Any]:
    """Преобразует одну сырую запись из data[] в плоский словарь с понятными именами колонок."""
    letter_meta = extract_letter_meta(row.get("col10"))
    return {
        "letter_id": letter_meta["letter_id"],
        "trade_name": extract_cell_value(row.get("col1"), prefer_title=True),
        "dosage_form_raw": extract_cell_value(row.get("col2"), prefer_title=True),
        "batch_number": extract_cell_value(row.get("col3"), prefer_title=True),
        "manufacturer_raw": extract_cell_value(row.get("col4"), prefer_title=True),
        "country": extract_cell_value(row.get("col5"), prefer_title=True),
        "status": extract_cell_value(row.get("col7"), prefer_title=True),
        "regulatory_action": extract_cell_value(row.get("col8"), prefer_title=True),
        "scope": extract_cell_value(row.get("col9"), prefer_title=True),
        "letter_raw": letter_meta["letter_raw"],
        "letter_link": letter_meta["letter_link"],
        "defect_parameter": extract_cell_value(row.get("col11"), prefer_title=True),
        "reg_number": extract_cell_value(row.get("col12"), prefer_title=True),
    }


class RoszdravnadzorScraper:
    """Клиент для выгрузки реестра изъятых из обращения ЛС с сайта Росздравнадзора."""

    def __init__(
        self,
        timeout: int = 30,
        max_retries: int = 5,
        backoff_factor: float = 1.0,
        sleep_between_requests: float = 0.4,
    ) -> None:
        self.timeout = timeout
        self.sleep_between_requests = sleep_between_requests
        self.verify_ssl = True
        self.session = self._build_session(max_retries=max_retries, backoff_factor=backoff_factor)

    def _build_session(self, max_retries: int, backoff_factor: float) -> requests.Session:
        session = requests.Session()
        session.headers.update(DEFAULT_HEADERS)
        retry_strategy = Retry(
            total=max_retries,
            connect=max_retries,
            read=max_retries,
            status=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def _request_post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Выполняет POST-запрос к API с автоматическим фоллбэком при ошибке сертификата Минцифры."""
        try:
            response = self.session.post(
                AJAX_API_URL,
                data=payload,
                timeout=self.timeout,
                verify=self.verify_ssl,
            )
        except requests.exceptions.SSLError:
            if self.verify_ssl:
                logger.warning(
                    "SSL сертификат roszdravnadzor.gov.ru (НУЦ Минцифры) не найден в локальном "
                    "хранилище certifi. Переключаемся на verify=False."
                )
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                self.verify_ssl = False
                response = self.session.post(
                    AJAX_API_URL,
                    data=payload,
                    timeout=self.timeout,
                    verify=False,
                )
            else:
                raise

        response.raise_for_status()
        return response.json()

    def fetch_page(
        self,
        start: int = 0,
        length: int = 100,
        draw: int = 1,
        prev_total: int = 0,
        date_from: str = "01.01.2020",
        date_to: str = "",
    ) -> dict[str, Any]:
        """Запрашивает одну страницу данных (по умолчанию 100 записей)."""
        payload = {
            "draw": draw,
            "start": start,
            "length": length,
            "prev_total": prev_total,
            "tn": "",
            "q_tn": "",
            "producer": "",
            "series": "",
            "status": "",
            "country": "",
            "let_from": date_from,
            "let_to": date_to,
            "q_mnn": "",
            "let_num": "",
        }
        return self._request_post(payload)

    def fetch_records(
        self,
        date_from: str = "01.01.2020",
        date_to: str = "",
        page_size: int = 100,
        max_records: int | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Выгружает записи из API Росздравнадзора с пагинацией.
        Возвращает кортеж: (список очищенных словарей, список сырых объектов data[]).
        """
        parsed_rows: list[dict[str, Any]] = []
        raw_rows: list[dict[str, Any]] = []

        start = 0
        draw = 1
        prev_total = 0
        total_available: int | None = None

        logger.info(
            "Старт выгрузки данных Росздравнадзора (let_from=%s, let_to=%s, max_records=%s)",
            date_from,
            date_to or "сегодня",
            max_records or "все",
        )

        while True:
            current_length = page_size
            if max_records is not None:
                remaining = max_records - len(parsed_rows)
                if remaining <= 0:
                    break
                current_length = min(page_size, remaining)

            data_json = self.fetch_page(
                start=start,
                length=current_length,
                draw=draw,
                prev_total=prev_total,
                date_from=date_from,
                date_to=date_to,
            )

            if total_available is None:
                records_filtered = int(data_json.get("recordsFiltered") or 0)
                records_total = int(data_json.get("recordsTotal") or 0)
                total_available = records_filtered or records_total
                prev_total = total_available
                target_total = min(max_records, total_available) if max_records else total_available
                pbar = tqdm(total=target_total, desc="Выгрузка РЗН", unit="зап.")
                logger.info(
                    "Ответ API получен: всего записей по фильтру recordsFiltered=%d (recordsTotal=%d)",
                    records_filtered,
                    records_total,
                )

            batch = data_json.get("data") or []
            if not batch:
                logger.info("Пустой блок data[] на смещении start=%d. Выгрузка завершена.", start)
                break

            added_in_batch = 0
            for item in batch:
                raw_rows.append(item)
                parsed_rows.append(parse_record(item))
                added_in_batch += 1
                if max_records is not None and len(parsed_rows) >= max_records:
                    break

            if "pbar" in locals():
                pbar.update(added_in_batch)

            if max_records is not None and len(parsed_rows) >= max_records:
                break
            if total_available and len(parsed_rows) >= total_available:
                break

            start += len(batch)
            draw += 1
            time.sleep(self.sleep_between_requests)

        if "pbar" in locals():
            pbar.close()

        logger.info("Выгрузка успешно завершена: получено %d записей.", len(parsed_rows))
        return parsed_rows, raw_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Парсер изъятых из обращения лекарственных средств (Росздравнадзор)"
    )
    parser.add_argument(
        "--date-from",
        type=str,
        default="01.01.2020",
        help="Начальная дата письма в формате DD.MM.YYYY (по умолчанию: 01.01.2020)",
    )
    parser.add_argument(
        "--date-to",
        type=str,
        default="",
        help="Конечная дата письма в формате DD.MM.YYYY (по умолчанию: пусто)",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=100,
        help="Максимальное число записей для скачивания (0 — скачать все)",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=100,
        help="Размер страницы пагинации (по умолчанию: 100)",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="data/raw/sample_100.csv",
        help="Путь для сохранения CSV-файла",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="data/raw/sample_100.json",
        help="Путь для сохранения сырого JSON-ответа",
    )
    args = parser.parse_args()

    max_records = args.max_records if args.max_records > 0 else None

    scraper = RoszdravnadzorScraper()
    parsed_rows, raw_rows = scraper.fetch_records(
        date_from=args.date_from,
        date_to=args.date_to,
        page_size=args.page_size,
        max_records=max_records,
    )

    if not parsed_rows:
        logger.error("Не удалось получить записи из API Росздравнадзора.")
        sys.exit(1)

    df = pd.DataFrame(parsed_rows)

    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info("Сохранено %d строк в CSV: %s", len(df), csv_path)

    if args.output_json:
        json_path = Path(args.output_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(raw_rows, f, ensure_ascii=False, indent=2)
        logger.info("Сохранен сырой JSON (%d записей): %s", len(raw_rows), json_path)


if __name__ == "__main__":
    main()
