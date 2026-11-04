"""
Модуль очистки, доменного фичинга и расчета скоринговых метрик качества ЛС (PRRI, TDR, Series-per-Incident).

Основные задачи:
1. Парсинг номера и даты информационного письма Росздравнадзора из col10.
2. Нормализация названий фармпроизводителей и выделение канонического имени завода.
3. Классификация лекарственных форм из col2 по технологическим группам (ТЛФ, Стерильные/Инъекции, ЖЛФ, МЛФ, АФС).
4. Доменная классификация показателей брака (col11) по 3 уровням критичности (Packaging, Technological, Critical API/Micro).
5. Векторный расчет индекса PRRI (Pharma Recall Risk Index), TDR (%) и Series-per-Incident Ratio.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("rzn_cleaner")

# Регулярные выражения для парсинга писем Росздравнадзора (например: "№ 01И-1279/26 от 03.11.2026 799 Кб")
LETTER_NUM_RE = re.compile(r"№\s*([^\s]+)")
LETTER_DATE_RE = re.compile(r"от\s*(\d{2}\.\d{2}\.\d{4})")
PAREN_SHORT_RE = re.compile(r"\(([^()]+)\)\s*$")

# Канонический словарь объединения дублей производителей (смена ОПФ ОАО -> ПАО/АО, ООО -> АО и т.д.)
MANUFACTURER_ALIASES: dict[str, str] = {
    'ООО "ВЕРОФАРМ"': 'АО "ВЕРОФАРМ"',
    'АО "ВЕРОФАРМ"': 'АО "ВЕРОФАРМ"',
    'ОАО "Синтез"': 'ПАО "Синтез"',
    'ПАО "Синтез"': 'ПАО "Синтез"',
    'ОАО "Акционерное Курганское общество медицинских препаратов и изделий "Синтез"': 'ПАО "Синтез"',
    'ОАО "Биосинтез"': 'ПАО "Биосинтез"',
    'ПАО "Биосинтез"': 'ПАО "Биосинтез"',
    'ОАО "Красфарма"': 'ПАО "Красфарма"',
    'ПАО "Красфарма"': 'ПАО "Красфарма"',
    'ОАО "Ирбитский химфармзавод"': 'АО "Ирбитский химфармзавод"',
    'АО "Ирбитский химфармзавод"': 'АО "Ирбитский химфармзавод"',
    'ОАО "ДАЛЬХИМФАРМ"': 'ОАО "ДАЛЬХИМФАРМ"',
    'ОАО "Дальхимфарм"': 'ОАО "ДАЛЬХИМФАРМ"',
    'ОАО "Самарамедпром"': 'ОАО "Самарамедпром"',
    'ООО "Пранафарм"': 'ООО "ПРАНАФАРМ"',
    'ООО "ПРАНАФАРМ"': 'ООО "ПРАНАФАРМ"',
    'АО "Рафарма"': 'АО "Рафарма"',
    'Акционерное Общество "Рафарма"': 'АО "Рафарма"',
    'Хемофарм А.Д.': 'ООО "Хемофарм" / Hemofarm',
    'ООО "Хемофарм"': 'ООО "Хемофарм" / Hemofarm',
}

# Замена полных организационно-правовых форм на стандартные аббревиатуры
LEGAL_FORM_REPLACEMENTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^Общество с ограниченной ответственностью\s+", re.IGNORECASE), "ООО "),
    (re.compile(r"^Публичное акционерное общество\s+", re.IGNORECASE), "ПАО "),
    (re.compile(r"^Открытое акционерное общество\s+", re.IGNORECASE), "ОАО "),
    (re.compile(r"^Закрытое акционерное общество\s+", re.IGNORECASE), "ЗАО "),
    (re.compile(r"^Акционерное общество\s+", re.IGNORECASE), "АО "),
    (re.compile(r"^Федеральное государственное унитарное предприятие\s+", re.IGNORECASE), "ФГУП "),
    (re.compile(r"^Федеральное казенное предприятие\s+", re.IGNORECASE), "ФКП "),
    (re.compile(r"^Федеральное государственное бюджетное учреждение\s+", re.IGNORECASE), "ФГБУ "),
]


def parse_letter_info(df: pd.DataFrame) -> pd.DataFrame:
    """Извлекает номер письма, дату письма и рассчитывает временные признаки."""
    raw_series = df["letter_raw"].fillna("").astype(str)
    df["letter_number"] = raw_series.str.extract(LETTER_NUM_RE, expand=False).fillna("Не указан")
    date_str = raw_series.str.extract(LETTER_DATE_RE, expand=False)
    df["letter_date"] = pd.to_datetime(date_str, format="%d.%m.%Y", errors="coerce")

    # Если у единичных записей дата не распарсилась, заполняем медианой или ближайшей датой
    if df["letter_date"].isna().any():
        df["letter_date"] = df["letter_date"].ffill().bfill()

    df["year"] = df["letter_date"].dt.year.astype(int)
    df["month"] = df["letter_date"].dt.month.astype(int)
    df["quarter"] = df["letter_date"].dt.quarter.astype(int)
    df["year_month"] = df["letter_date"].dt.strftime("%Y-%m")

    ref_date = df["letter_date"].max()
    df["age_days"] = (ref_date - df["letter_date"]).dt.days.clip(lower=0)
    df["age_years"] = (df["age_days"] / 365.25).round(4)
    return df


def normalize_manufacturer_name(raw_name: str | float) -> str:
    """Приводит название производителя к каноническому короткому виду."""
    if pd.isna(raw_name) or not str(raw_name).strip():
        return "Не указан"

    text = str(raw_name).strip()
    text = text.replace("«", '"').replace("»", '"').replace("“", '"').replace("”", '"')
    text = re.sub(r"\s+", " ", text)

    if "отсутствует маркировка" in text.lower():
        return "Не установлен (фальсификат / без маркировки)"

    # Если в конце в скобках указано сокращенное юр. лицо, например: ...(ООО "Велфарм-М")
    match = PAREN_SHORT_RE.search(text)
    if match:
        candidate = match.group(1).strip()
        upper_cand = candidate.upper()
        if any(
            token in upper_cand
            for token in ('"', "ООО", "АО", "ПАО", "ОАО", "ЗАО", "ФГУП", "ФКП", "ФГБУ", "НПО", "ГУП")
        ):
            text = candidate

    for pattern, repl in LEGAL_FORM_REPLACEMENTS:
        text = pattern.sub(repl, text)

    text = re.sub(r"\s+", " ", text).strip()
    return MANUFACTURER_ALIASES.get(text, text)


def classify_dosage_form(raw_form: str | float) -> tuple[str, str]:
    """
    Возвращает кортеж (dosage_form_group, dosage_form_short) на основе строки лекарственной формы.
    """
    if pd.isna(raw_form) or not str(raw_form).strip():
        return ("Прочие / АФС", "Не указана")

    s = str(raw_form).strip().lower()

    # 1. Субстанции и фармсырье
    if "субстанция" in s or "сырье растительное" in s or s.startswith("трава ") or s.startswith("листья "):
        return ("АФС и фармсырье", "Субстанция / Сырье")

    # 2. Стерильные и парентеральные формы (инъекции, инфузии, лиофилизаты, глазные капли)
    is_parenteral_or_sterile = any(
        kw in s
        for kw in (
            "внутривенн",
            "внутримышечн",
            "подкожн",
            "инъекц",
            "инфуз",
            "парентеральн",
            "лиофилизат",
            "глазные",
            "глазная",
            "растворитель для приготовления лекарственных форм для инъекций",
            "имплантат",
        )
    )
    if is_parenteral_or_sterile:
        if "лиофилизат" in s or "порошок" in s:
            return ("Стерильные и парентеральные ЛФ", "Лиофилизат / Порошок для инъекций")
        if "концентрат" in s:
            return ("Стерильные и парентеральные ЛФ", "Концентрат для инфузий/инъекций")
        if "глазн" in s:
            return ("Стерильные и парентеральные ЛФ", "Капли / Формы глазные")
        return ("Стерильные и парентеральные ЛФ", "Раствор для инъекций/инфузий")

    # 3. ТЛФ (Твердые лекарственные формы)
    if s.startswith("таблетки"):
        if "кишечнорастворим" in s or "пролонгирован" in s or "модифицирован" in s:
            return ("ТЛФ (Твердые лек. формы)", "Таблетки с модиф./кишечнораств. оболочкой")
        if "оболочк" in s:
            return ("ТЛФ (Твердые лек. формы)", "Таблетки, покрытые оболочкой")
        return ("ТЛФ (Твердые лек. формы)", "Таблетки непокрытые")
    if s.startswith("капсулы"):
        return ("ТЛФ (Твердые лек. формы)", "Капсулы")
    if s.startswith("драже") or s.startswith("пастилки") or s.startswith("леденцы"):
        return ("ТЛФ (Твердые лек. формы)", "Драже / Пастилки")
    if s.startswith("порошок") or s.startswith("гранулы") or s.startswith("сбор"):
        return ("ТЛФ (Твердые лек. формы)", "Порошки и гранулы (пероральные/наружные)")

    # 4. МЛФ (Мягкие лекарственные формы)
    if any(s.startswith(prefix) for prefix in ("мазь", "крем", "гель", "линимент", "паста")):
        return ("МЛФ (Мягкие лек. формы)", "Мазь / Крем / Гель")
    if s.startswith("суппозитории"):
        return ("МЛФ (Мягкие лек. формы)", "Суппозитории")

    # 5. ЖЛФ нестерильные (пероральные, наружные, ингаляционные жидкости)
    if any(
        s.startswith(prefix)
        for prefix in (
            "раствор",
            "концентрат",
            "капли",
            "суспензия",
            "сироп",
            "спрей",
            "аэрозоль",
            "настойка",
            "экстракт",
            "эмульсия",
            "эликсир",
            "бальзам",
            "растворитель",
            "масло",
            "шампунь",
            "жидкость",
        )
    ):
        if s.startswith("суспензия") or s.startswith("эмульсия") or s.startswith("сироп"):
            return ("ЖЛФ нестерильные", "Суспензия / Сироп / Эмульсия")
        if s.startswith("капли") or s.startswith("спрей") or s.startswith("аэрозоль"):
            return ("ЖЛФ нестерильные", "Капли / Спрей / Аэрозоль")
        if s.startswith("настойка") or s.startswith("экстракт") or s.startswith("бальзам") or s.startswith("эликсир"):
            return ("ЖЛФ нестерильные", "Галеновые формы (настойка/экстракт)")
        return ("ЖЛФ нестерильные", "Раствор (пероральный / наружный)")

    return ("Прочие / АФС", "Прочие лекарственные формы")


def classify_single_defect(token: str) -> tuple[int, float, str]:
    """
    Классифицирует один атомарный показатель брака из col11 по 3 уровням критичности:
    Возвращает: (level_int, weight_float, defect_category_str)
      - Level 3 (5.0): Critical Safety, API & Microbiology
      - Level 2 (2.5): Technological & Pharmacopoeial Form
      - Level 1 (1.0): Packaging & Labeling
    """
    t = token.strip().lower()
    if not t:
        return (0, 1.5, "Превентивный отзыв (показатель не указан)")

    # --- LEVEL 3: CRITICAL SAFETY, API & MICROBIOLOGY (вес 5.0) ---
    # 3.1 Микробиология, стерильность, эндотоксины, пирогенность, токсичность
    if any(
        kw in t
        for kw in (
            "микробиолог",
            "микроорганизм",
            "стерильност",
            "эндотоксин",
            "пироген",
            "токсичност",
            "вирус",
            "антитела",
        )
    ):
        return (3, 5.0, "Микробиология, стерильность и эндотоксины")

    # 3.2 Примеси, продукты деградации АФС и остаточные растворители
    if any(
        kw in t
        for kw in (
            "примес",
            "дисульфид",
            "салициловая кислота",
            "растворител",
            "триэтиламин",
            "пирролидин",
            "хлоранилин",
            "трибромфенол",
            "альдегид",
            "полимеры и агрегаты",
            "алюминий",
            "спирт метиловый",
            "каприловая кислота",
        )
    ):
        return (3, 5.0, "Родственные и посторонние примеси (стабильность АФС)")

    # 3.3 Количественное определение, дозировка АФС, специфическая активность
    if any(
        kw in t
        for kw in (
            "количественн",
            "активност",
            "ацетилтриптофан",
            "стабилизатор",
            "общий белок",
            "состав белка",
            "альбумин",
            "белок",
            "глицин",
            "цитрат",
            "кальция хлорид",
            "натрия хлорид",
            "натрия цитрат",
            "трисгидроксиметиламинометан",
            "калий",
            "кальций",
            "натрий",
        )
    ):
        return (3, 5.0, "Количественное определение и содержание АФС")

    # 3.4 Подлинность и идентификация
    if any(
        kw in t
        for kw in (
            "подлинност",
            "идентификац",
            "качественные реакции",
            "состав",
        )
    ):
        return (3, 5.0, "Подлинность и идентификация")

    # --- LEVEL 1: PACKAGING & LABELING (вес 1.0) ---
    if any(
        kw in t
        for kw in (
            "маркировк",
            "упаковк",
            "комплектност",
            "срок годности",
            "инструкц",
        )
    ):
        # Важно: "Объем содержимого упаковки" и "Масса содержимого упаковки" — это дозирование/фасовка (Level 2)
        if "содержимого упаковки" not in t:
            return (1, 1.0, "Маркировка и упаковка")

    # --- LEVEL 2: TECHNOLOGICAL & PHARMACOPOEIAL FORM (вес 2.5) ---
    # 2.1 Растворение и распадаемость (биофармацевтика ТЛФ)
    if any(
        kw in t
        for kw in (
            "растворен",
            "распадаемост",
            "полной деформации",
            "растворимост",
        )
    ):
        return (2, 2.5, "Растворение и распадаемость")

    # 2.2 Однородность дозирования, однородность массы, средняя масса, объем наполнения
    if any(
        kw in t
        for kw in (
            "однородност",
            "средняя масса",
            "масса содержимого",
            "объем содержимого",
            "объём содержимого",
            "извлекаемый объем",
            "истираемост",
        )
    ):
        return (2, 2.5, "Однородность дозирования и массы")

    # 2.3 Механические включения (парентеральные и жидкие формы)
    if "механические включения" in t:
        return (2, 2.5, "Механические включения")

    # 2.4 Описание, цветность, прозрачность, внешний вид (органолептика и стабильность формы)
    if any(
        kw in t
        for kw in (
            "описание",
            "внешний вид",
            "цветност",
            "прозрачност",
            "мутност",
            "термостабильност",
        )
    ):
        return (2, 2.5, "Описание, цветность и прозрачность")

    # 2.5 Физико-химические параметры формы (pH, вода, потеря в массе, плотность, осмоляльность)
    if any(
        kw in t
        for kw in (
            "рн",
            "ph",
            "кислотно",
            "щелочност",
            "вода",
            "высушиван",
            "плотност",
            "вязкост",
            "осмоляльност",
            "размер частиц",
            "спирт этиловый",
            "молекулярные параметры",
        )
    ):
        return (2, 2.5, "Физико-химические параметры (pH, влага, осмоляльность)")

    # Фоллбэк для редких фармакопейных показателей
    return (2, 2.5, "Физико-химические параметры (pH, влага, осмоляльность)")


LEVEL_LABELS: dict[int, str] = {
    0: "Level 0: Превентивный отзыв (без показателя в реестре)",
    1: "Level 1: Логистика и упаковка (1.0)",
    2: "Level 2: Технологический брак формы (2.5)",
    3: "Level 3: Критический брак АФС и микробиологии (5.0)",
}


def classify_row_defect(defect_raw: str | float, status: str) -> tuple[int, str, float, str, int, bool]:
    """
    Классифицирует строку инцидента с учетом комбинированных показателей (через ';') и статуса ЛС.
    Возвращает:
      (defect_level_num, defect_level_label, w_defect, defect_category, defect_params_count, has_explicit_defect)
    """
    status_lower = str(status or "").strip().lower()

    # Если показатель брака в col11 пуст
    if pd.isna(defect_raw) or not str(defect_raw).strip():
        if any(
            k in status_lower
            for k in ("фальсифиц", "незаконный оборот", "незарегистрирован")
        ):
            return (
                3,
                LEVEL_LABELS[3],
                5.0,
                "Фальсификат / Незаконный оборот",
                1,
                True,
            )
        return (
            0,
            LEVEL_LABELS[0],
            1.5,
            "Превентивный отзыв (показатель не указан)",
            0,
            False,
        )

    tokens = [tok.strip() for tok in str(defect_raw).split(";") if tok.strip()]
    if not tokens:
        return (
            0,
            LEVEL_LABELS[0],
            1.5,
            "Превентивный отзыв (показатель не указан)",
            0,
            False,
        )

    # Оцениваем каждый показатель в строке и берем дефект с МАКСИМАЛЬНОЙ критичностью
    classified = [classify_single_defect(tok) for tok in tokens]
    best_level, best_weight, best_cat = max(classified, key=lambda x: (x[0], x[1]))
    return (
        best_level,
        LEVEL_LABELS[best_level],
        best_weight,
        best_cat,
        len(tokens),
        True,
    )


def compute_action_weight(action: str | float) -> tuple[float, bool]:
    """
    Определяет вес строгости регуляторной меры W_action и флаг is_recall_event.
    - Изъятие / Отзыв / Прекращение обращения -> 1.5
    - Приостановление реализации / Информационное -> 1.0
    - Разрешение реализации (снятие карантина после проверки) -> 0.0 (не является браком!)
    """
    s = str(action or "").strip().lower()
    if "разрешение" in s:
        return (0.0, False)
    if any(k in s for k in ("изъятие", "отзыв", "прекращение")):
        return (1.5, True)
    return (1.0, True)


def build_cleaned_dataset(
    raw_csv_path: str | Path = "data/raw/rzn_full_raw.csv",
    cleaned_csv_path: str | Path = "data/processed/rzn_cleaned.csv",
    mfg_csv_path: str | Path = "data/processed/manufacturers_prri.csv",
    decay_lambda: float = 0.50,
    chronic_alpha: float = 0.25,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Полный пайплайн очистки данных Росздравнадзора и расчета индекса PRRI.
    """
    logger.info("Чтение сырого датасета: %s", raw_csv_path)
    df = pd.read_csv(raw_csv_path)
    initial_len = len(df)

    # Удаляем полные технические дубликаты, если они возникли на стыке пагинации
    df = df.drop_duplicates(
        subset=["letter_raw", "trade_name", "batch_number", "manufacturer_raw", "defect_parameter"]
    ).copy()
    logger.info("Строк после удаления полных дубликатов: %d (из %d)", len(df), initial_len)

    # 1. Парсинг писем и дат
    df = parse_letter_info(df)

    # 2. Нормализация производителей и страны
    df["country"] = df["country"].fillna("Не указана").astype(str).str.strip()
    df["country"] = df["country"].replace({"": "Не указана", "nan": "Не указана"})
    df["origin_type"] = np.where(
        df["country"].str.lower() == "россия",
        "Отечественный (РФ)",
        "Зарубежный (Импорт)",
    )
    df["manufacturer_clean"] = df["manufacturer_raw"].apply(normalize_manufacturer_name)

    # 3. Классификация лекарственных форм
    form_parsed = df["dosage_form_raw"].apply(classify_dosage_form)
    df["dosage_form_group"] = [item[0] for item in form_parsed]
    df["dosage_form_short"] = [item[1] for item in form_parsed]

    # 4. Доменная классификация показателей брака (Level 0..3)
    defect_parsed = [
        classify_row_defect(def_val, stat_val)
        for def_val, stat_val in zip(df["defect_parameter"], df["status"])
    ]
    df["defect_level_num"] = [item[0] for item in defect_parsed]
    df["defect_level"] = [item[1] for item in defect_parsed]
    df["w_defect"] = [item[2] for item in defect_parsed]
    df["defect_category"] = [item[3] for item in defect_parsed]
    df["defect_params_count"] = [item[4] for item in defect_parsed]
    df["has_explicit_defect"] = [item[5] for item in defect_parsed]

    # Флаг технологического/химико-микробиологического брака (Level 2 или Level 3)
    df["is_tech_or_critical_defect"] = df["defect_level_num"].isin([2, 3])

    # 5. Вес строгости регуляторной меры (W_action) и исключение «Разрешения реализации»
    action_parsed = df["regulatory_action"].apply(compute_action_weight)
    df["w_action"] = [item[0] for item in action_parsed]
    df["is_recall_event"] = [item[1] for item in action_parsed]

    # 6. Фактор системности / серийности (W_chronic)
    # Считаем число забракованных серий у каждого завода по каждой категории дефекта (среди реальных отзывов)
    recall_mask = df["is_recall_event"]
    mfg_cat_counts = (
        df[recall_mask]
        .groupby(["manufacturer_clean", "defect_category"])["batch_number"]
        .transform("count")
    )
    df["mfg_same_defect_batches"] = 0
    df.loc[recall_mask, "mfg_same_defect_batches"] = mfg_cat_counts

    # Повышающий коэффициент W_chronic применяется к технологическим и критическим дефектам (Level 2 и Level 3)
    # при числе серий > 1: W_chronic = 1 + alpha * ln(N)
    df["w_chronic"] = np.where(
        df["is_recall_event"] & df["is_tech_or_critical_defect"] & (df["mfg_same_defect_batches"] > 1),
        (1.0 + chronic_alpha * np.log(df["mfg_same_defect_batches"].clip(lower=1))).round(4),
        1.0,
    )

    # 7. Временное затухание (Recency Decay: e^(-lambda * max(0, age_years - 1.0)))
    # Свежие забраковки за последние 12 месяцев имеют максимальный вес 1.0, более старые затухают
    effective_age = (df["age_years"] - 1.0).clip(lower=0.0)
    df["recency_decay"] = np.exp(-decay_lambda * effective_age).round(4)

    # 8. Итоговый скоринг инцидента:
    # prri_incident_score — по верифицированным дефектам качества (Level 1-3)
    # prri_total_incident_score — включая превентивные отзывы без указанного показателя (Level 0-3)
    df["prri_total_incident_score"] = np.where(
        df["is_recall_event"],
        (df["w_defect"] * df["w_action"] * df["w_chronic"] * df["recency_decay"]).round(4),
        0.0,
    )
    df["prri_incident_score"] = np.where(
        df["is_recall_event"] & df["has_explicit_defect"],
        df["prri_total_incident_score"],
        0.0,
    )

    # Сохраняем очищенный датасет инцидентов
    cleaned_path = Path(cleaned_csv_path)
    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cleaned_path, index=False, encoding="utf-8-sig")
    logger.info("Сохранен очищенный датасет инцидентов (%d строк): %s", len(df), cleaned_path)

    # 9. Формируем агрегированную витрину скоринга производителей (manufacturers_prri.csv)
    recalls_df = df[df["is_recall_event"]].copy()
    mfg_summary = (
        recalls_df.groupby(["manufacturer_clean", "origin_type"], as_index=False)
        .agg(
            country=("country", "first"),
            total_recalls=("batch_number", "count"),
            unique_letters=("letter_number", "nunique"),
            unique_drugs=("trade_name", "nunique"),
            explicit_defects_count=("has_explicit_defect", "sum"),
            level1_packaging_count=("defect_level_num", lambda s: int((s == 1).sum())),
            level2_tech_count=("defect_level_num", lambda s: int((s == 2).sum())),
            level3_critical_count=("defect_level_num", lambda s: int((s == 3).sum())),
            level0_unspecified_count=("defect_level_num", lambda s: int((s == 0).sum())),
            recent_12m_recalls=("age_years", lambda s: int((s <= 1.0).sum())),
            prri_score=("prri_incident_score", lambda s: round(float(s.sum()), 2)),
            prri_total_score=("prri_total_incident_score", lambda s: round(float(s.sum()), 2)),
            avg_incident_severity=("w_defect", lambda s: round(float(s.mean()), 2)),
            max_chronic_factor=("w_chronic", lambda s: round(float(s.max()), 2)),
            last_recall_date=("letter_date", "max"),
        )
    )

    # Расчет дополнительных метрик:
    # 1) Series-per-Incident Ratio (среднее число серий на одно письмо РЗН)
    mfg_summary["series_per_incident"] = (
        mfg_summary["total_recalls"] / mfg_summary["unique_letters"].clip(lower=1)
    ).round(2)

    # 2) TDR (Technological Defect Share, %) — доля технологического и критического брака (Level 2 + Level 3)
    # среди всех верифицированных дефектов (Level 1 + Level 2 + Level 3)
    tech_crit_sum = mfg_summary["level2_tech_count"] + mfg_summary["level3_critical_count"]
    mfg_summary["tdr_pct"] = np.where(
        mfg_summary["explicit_defects_count"] > 0,
        (tech_crit_sum / mfg_summary["explicit_defects_count"] * 100.0).round(1),
        0.0,
    )

    # Классификация уровня риска поставщика (Risk Tier) на пересечении PRRI и доли тех. брака (TDR %)
    def _classify_tier(row: pd.Series) -> str:
        score = float(row["prri_score"])
        tdr = float(row["tdr_pct"])
        if score < 15.0:
            return "Low Risk (Низкий / единичный риск)"
        if tdr < 40.0:
            return "Packaging Risk (Преимущественно упаковка/маркировка)"
        if score >= 150.0:
            return "Critical Risk (Критический тех. риск)"
        if score >= 50.0:
            return "High Risk (Системный тех. брак)"
        return "Moderate Risk (Умеренный тех. риск)"

    mfg_summary["risk_tier"] = mfg_summary.apply(_classify_tier, axis=1)

    mfg_summary = mfg_summary.sort_values(
        by=["prri_score", "prri_total_score"], ascending=False
    ).reset_index(drop=True)

    mfg_path = Path(mfg_csv_path)
    mfg_path.parent.mkdir(parents=True, exist_ok=True)
    mfg_summary.to_csv(mfg_path, index=False, encoding="utf-8-sig")
    logger.info("Сохранен рейтинг производителей (%d заводов): %s", len(mfg_summary), mfg_path)

    return df, mfg_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Очистка данных Росздравнадзора и расчет скорингового индекса PRRI"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/rzn_full_raw.csv",
        help="Путь к сырому CSV-файлу",
    )
    parser.add_argument(
        "--output-cleaned",
        type=str,
        default="data/processed/rzn_cleaned.csv",
        help="Путь для сохранения очищенного датасета инцидентов",
    )
    parser.add_argument(
        "--output-mfg",
        type=str,
        default="data/processed/manufacturers_prri.csv",
        help="Путь для сохранения витрины производителей с PRRI",
    )
    args = parser.parse_args()

    build_cleaned_dataset(
        raw_csv_path=args.input,
        cleaned_csv_path=args.output_cleaned,
        mfg_csv_path=args.output_mfg,
    )


if __name__ == "__main__":
    main()
