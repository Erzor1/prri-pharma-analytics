"""
Интерактивный аналитический веб-дашборд (Streamlit + Plotly):
«Мониторинг и скоринг технологического брака, отзывов и стабильности ЛС на рынке РФ (PRRI)»
На живых данных Росздравнадзора (2020–2026).
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.cleaner import build_cleaned_dataset
from src.database import get_db_engine, init_database, query_df
from src.scraper import RoszdravnadzorScraper
from src.stats_tests import run_all_hypothesis_tests

st.set_page_config(
    page_title="PRRI — Мониторинг и скоринг брака ЛС (Росздравнадзор)",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Кастомные стили для карточек KPI (без обрезки текста и без ложных зеленых стрелок st.metric)
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }
    .kpi-card {
        background: rgba(28, 131, 225, 0.05);
        border: 1px solid rgba(28, 131, 225, 0.22);
        border-radius: 12px;
        padding: 14px 16px;
        height: 100%;
        min-height: 118px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .kpi-title {
        font-size: 0.86rem;
        font-weight: 600;
        color: #4a5568;
        line-height: 1.25;
        margin-bottom: 6px;
        white-space: normal;
        word-wrap: break-word;
    }
    .kpi-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #1a202c;
        line-height: 1.15;
        margin-bottom: 6px;
    }
    .kpi-sub {
        font-size: 0.80rem;
        color: #2b6cb0;
        background: rgba(43, 108, 176, 0.10);
        padding: 4px 8px;
        border-radius: 6px;
        line-height: 1.3;
        white-space: normal;
        word-wrap: break-word;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_kpi_card(title: str, value: str, subtitle: str) -> None:
    """Рендерит адаптивную карточку KPI, где весь текст виден полностью без троеточий и стрелок."""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


RAW_CSV_PATH = Path("data/raw/rzn_full_raw.csv")
CLEANED_CSV_PATH = Path("data/processed/rzn_cleaned.csv")
MFG_CSV_PATH = Path("data/processed/manufacturers_prri.csv")
SYNC_META_PATH = Path("data/processed/sync_meta.json")
DB_MANIFEST_PATH = Path("data/processed/.db_manifest.json")
BACKUP_DIR = Path("data/processed/.backup")

# Streamlit Community Cloud разворачивает приложение из каталога /mount/src.
# В облаке файловая система контейнера одноразовая и доступ к .gov.ru нестабилен,
# поэтому «живая синхронизация» там отключена (кнопка-заглушка).
IS_CLOUD = Path("/mount/src").exists() or os.getenv("STREAMLIT_CLOUD", "").strip() == "1"


def count_csv_rows(path: Path) -> int:
    """Быстро считает число строк данных в CSV без его полной загрузки в память."""
    try:
        with path.open("r", encoding="utf-8-sig", errors="ignore") as fh:
            return max(0, sum(1 for _ in fh) - 1)
    except OSError:
        return 0


def dataset_fingerprint(*paths: Path) -> str:
    """Контрольная сумма CSV-снимков: позволяет понять, что БД устарела и её надо пересобрать."""
    digest = hashlib.sha256()
    for path in paths:
        try:
            with path.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    digest.update(chunk)
        except OSError:
            return ""
    return digest.hexdigest()[:16]


def read_db_manifest() -> dict:
    try:
        return json.loads(DB_MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def rebuild_database() -> None:
    """Перезаливает факты и витрины в СУБД и фиксирует отпечаток CSV-снимков."""
    init_database(cleaned_csv_path=CLEANED_CSV_PATH, mfg_csv_path=MFG_CSV_PATH)
    manifest = {
        "fingerprint": dataset_fingerprint(CLEANED_CSV_PATH, MFG_CSV_PATH),
        "built_at": datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
        "cleaned_rows": count_csv_rows(CLEANED_CSV_PATH),
        "mfg_rows": count_csv_rows(MFG_CSV_PATH),
    }
    DB_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    DB_MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def ensure_data_ready() -> None:
    """
    Готовит данные к показу:
    1) при первом старте собирает очищенный датасет из сырой выгрузки;
    2) пересобирает БД и SQL-витрины, если CSV-снимки изменились (например, после git push
       на Streamlit Cloud, где файл БД не хранится в репозитории и может быть устаревшим).
    """
    if not RAW_CSV_PATH.exists():
        sample_path = Path("data/raw/sample_100.csv")
        if sample_path.exists() and not CLEANED_CSV_PATH.exists():
            with st.spinner("Первичная выгрузка реестра Росздравнадзора (2020–2026)..."):
                scraper = RoszdravnadzorScraper()
                parsed_rows, _ = scraper.fetch_records(date_from="01.01.2020", page_size=500, max_records=None)
                RAW_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(parsed_rows).to_csv(RAW_CSV_PATH, index=False, encoding="utf-8-sig")

    if RAW_CSV_PATH.exists() and (not CLEANED_CSV_PATH.exists() or not MFG_CSV_PATH.exists()):
        with st.spinner("Очистка данных и расчет индекса PRRI..."):
            build_cleaned_dataset(
                raw_csv_path=RAW_CSV_PATH,
                cleaned_csv_path=CLEANED_CSV_PATH,
                mfg_csv_path=MFG_CSV_PATH,
            )

    if not CLEANED_CSV_PATH.exists() or not MFG_CSV_PATH.exists():
        return

    engine = get_db_engine()
    try:
        query_df("SELECT 1 FROM v_manufacturer_risk_ranking LIMIT 1", engine=engine)
        views_ready = True
    except Exception:
        views_ready = False

    fingerprint = dataset_fingerprint(CLEANED_CSV_PATH, MFG_CSV_PATH)
    if views_ready and fingerprint and read_db_manifest().get("fingerprint") == fingerprint:
        return

    with st.spinner("Инициализация реляционной БД и SQL-витрин..."):
        rebuild_database()


@st.cache_data(ttl=600, show_spinner=False)
def load_dashboard_tables() -> dict[str, pd.DataFrame]:
    """Загружает основную таблицу фактов и 4 аналитические SQL-витрины."""
    ensure_data_ready()
    engine = get_db_engine()
    return {
        "fact_recalls": query_df("SELECT * FROM fact_recalls", engine=engine),
        "v_monthly": query_df("SELECT * FROM v_monthly_defect_dynamics ORDER BY year_month", engine=engine),
        "v_mfg_rank": query_df("SELECT * FROM v_manufacturer_risk_ranking", engine=engine),
        "v_form_matrix": query_df("SELECT * FROM v_dosage_form_defect_matrix", engine=engine),
        "v_chronic": query_df("SELECT * FROM v_chronic_drug_recidivism", engine=engine),
        "db_dialect": pd.DataFrame([{"dialect": engine.dialect.name.upper()}]),
    }


@st.cache_data(ttl=600, show_spinner=False)
def load_hypothesis_results() -> dict:
    engine = get_db_engine()
    return run_all_hypothesis_tests(engine=engine)


SYNC_META_PATH = Path("data/processed/sync_meta.json")


def get_sync_meta() -> dict[str, str]:
    """Читает метаданные последней успешной синхронизации с API РЗН (создаются локально)."""
    if SYNC_META_PATH.exists():
        try:
            meta = json.loads(SYNC_META_PATH.read_text(encoding="utf-8"))
            if isinstance(meta, dict):
                return meta
        except Exception:
            pass
    return {}


def _sync_keyset(df: pd.DataFrame) -> set[tuple]:
    """Ключ записи реестра РЗН для дедупликации при слиянии выгрузок."""
    key_cols = ["letter_raw", "trade_name", "batch_number", "manufacturer_raw", "defect_parameter"]
    if df.empty or any(col not in df.columns for col in key_cols):
        return set()
    return set(zip(*(df[col].fillna("").astype(str) for col in key_cols)))


def _write_sync_meta(meta: dict[str, str]) -> dict[str, str]:
    SYNC_META_PATH.parent.mkdir(parents=True, exist_ok=True)
    SYNC_META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def _restore_backup() -> None:
    """Откатывает CSV-снимки и БД из резервной копии, если пересборка потеряла историю."""
    for name, target in (
        ("rzn_full_raw.csv", RAW_CSV_PATH),
        ("rzn_cleaned.csv", CLEANED_CSV_PATH),
        ("manufacturers_prri.csv", MFG_CSV_PATH),
    ):
        backup_file = BACKUP_DIR / name
        if backup_file.exists():
            shutil.copy2(backup_file, target)
    rebuild_database()
    st.cache_data.clear()


def trigger_incremental_update() -> dict[str, str]:
    """
    Догружает свежие записи из API Росздравнадзора и пересчитывает метрики и БД.

    Запускается только в локальной версии дашборда. Ключевая защита: если полной выгрузки
    (2020+) нет, запрашивается весь период, а результат сверяется с предыдущим срезом —
    чтобы частичная выгрузка не «затёрла» историю данных.
    """
    date_from = "01.01.2026" if RAW_CSV_PATH.exists() else "01.01.2020"
    now_str = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
    rows_before = count_csv_rows(CLEANED_CSV_PATH)

    # Резервная копия перед любыми изменениями файлов
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    for source in (RAW_CSV_PATH, CLEANED_CSV_PATH, MFG_CSV_PATH):
        if source.exists():
            shutil.copy2(source, BACKUP_DIR / source.name)

    try:
        scraper = RoszdravnadzorScraper(timeout=20, max_retries=3)
        fresh_rows, _ = scraper.fetch_records(date_from=date_from, page_size=500, max_records=None)
        df_fresh = pd.DataFrame(fresh_rows)
    except Exception as exc:
        return _write_sync_meta(
            {
                "last_sync_ts": now_str,
                "latest_letter_date": _data_snapshot_date(),
                "latest_letter_raw": f"нет связи с API РЗН ({exc.__class__.__name__}) — показан сохранённый срез",
                "checked_records": "0",
                "added_records": "0",
                "status": "api_unavailable",
            }
        )

    if df_fresh.empty:
        return _write_sync_meta(
            {
                "last_sync_ts": now_str,
                "latest_letter_date": _data_snapshot_date(),
                "latest_letter_raw": "API РЗН не вернуло записей — сохранён прежний срез",
                "checked_records": "0",
                "added_records": "0",
                "status": "empty_response",
            }
        )

    if RAW_CSV_PATH.exists():
        df_old = pd.read_csv(RAW_CSV_PATH)
        added = len(_sync_keyset(df_fresh) - _sync_keyset(df_old))
        df_combined = pd.concat([df_fresh, df_old], ignore_index=True)
    else:
        added = len(df_fresh)
        df_combined = df_fresh

    df_combined = df_combined.drop_duplicates(
        subset=["letter_raw", "trade_name", "batch_number", "manufacturer_raw", "defect_parameter"]
    )
    RAW_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_combined.to_csv(RAW_CSV_PATH, index=False, encoding="utf-8-sig")

    df_cleaned, _ = build_cleaned_dataset(
        raw_csv_path=RAW_CSV_PATH,
        cleaned_csv_path=CLEANED_CSV_PATH,
        mfg_csv_path=MFG_CSV_PATH,
    )

    if rows_before > 1000 and len(df_cleaned) < rows_before * 0.9:
        _restore_backup()
        raise RuntimeError(
            f"Пересборка отменена: получено {len(df_cleaned)} строк вместо {rows_before}. "
            "Срез восстановлен из резервной копии — проверьте полноту выгрузки API."
        )

    rebuild_database()
    st.cache_data.clear()

    latest_row = df_cleaned.loc[pd.to_datetime(df_cleaned["letter_date"]).idxmax()]
    latest_letter = str(latest_row.get("letter_raw") or latest_row.get("letter_number") or "").strip()

    return _write_sync_meta(
        {
            "last_sync_ts": now_str,
            "latest_letter_date": str(pd.to_datetime(latest_row["letter_date"]).strftime("%Y-%m-%d")),
            "latest_letter_raw": latest_letter,
            "checked_records": str(len(df_fresh)),
            "added_records": str(added),
            "status": "ok",
        }
    )


def _data_snapshot_date() -> str:
    """Дата последнего письма РЗН в текущем срезе данных."""
    try:
        df = pd.read_csv(CLEANED_CSV_PATH, usecols=["letter_date"])
        return str(pd.to_datetime(df["letter_date"]).max().strftime("%Y-%m-%d"))
    except Exception:
        return "—"


def main() -> None:
    data = load_dashboard_tables()
    df_all = data["fact_recalls"]
    df_monthly = data["v_monthly"]
    df_mfg_rank = data["v_mfg_rank"]
    df_form_matrix = data["v_form_matrix"]
    df_chronic = data["v_chronic"]
    db_dialect = data["db_dialect"]["dialect"].iloc[0]

    if df_all.empty:
        st.error(
            "Не удалось загрузить данные Росздравнадзора: таблица фактов пуста. "
            "Проверьте наличие и корректность файлов data/processed/rzn_cleaned.csv и manufacturers_prri.csv."
        )
        st.stop()

    # -------------------------------------------------------------------------
    # SIDEBAR: Панель управления, фильтры и синхронизация с API Росздравнадзора
    # -------------------------------------------------------------------------
    sync_meta = get_sync_meta()
    years_available = sorted(
        {int(year) for year in pd.to_numeric(df_all["year"], errors="coerce").dropna().unique()}
    )
    min_year, max_year = years_available[0], years_available[-1]
    last_letter_dt = pd.to_datetime(df_all["letter_date"], errors="coerce").max()
    data_snapshot = "—" if pd.isna(last_letter_dt) else last_letter_dt.strftime("%d.%m.%Y")
    rows_total = f"{len(df_all):,}".replace(",", " ")
    api_checked = sync_meta.get("last_sync_ts") or "—"

    with st.sidebar:
        st.markdown("## 💊 PRRI Analytics")
        st.caption(
            "Мониторинг и скоринг технологического брака, отзывов и стабильности ЛС на рынке РФ"
        )
        mode_line = (
            "**Режим:** `демо` — срез данных зафиксирован в репозитории"
            if IS_CLOUD
            else f"**Проверка API РЗН:** `{api_checked}`"
        )
        st.markdown(
            f"**СУБД:** `{db_dialect}` | **Источник:** `roszdravnadzor.gov.ru`  \n"
            f"**Срез данных:** `{data_snapshot}` · `{rows_total}` записей ({min_year}–{max_year})  \n"
            f"{mode_line}"
        )
        st.divider()

        st.markdown("### 🔍 Глобальные фильтры")
        if min_year < max_year:
            selected_years = st.slider(
                "Период писем Росздравнадзора (годы):",
                min_value=min_year,
                max_value=max_year,
                value=(min_year, max_year),
            )
        else:
            # Защита от падения: st.slider требует min_value < max_value
            selected_years = (min_year, max_year)
            st.caption(f"В текущем срезе данных доступен только {min_year} год.")

        origin_options = ["Все площадки", "Отечественный (РФ)", "Зарубежный (Импорт)"]
        selected_origin = st.selectbox("Локализация производства:", origin_options, index=0)

        all_form_groups = sorted(df_all["dosage_form_group"].dropna().unique().tolist())
        selected_forms = st.multiselect(
            "Технологические группы ЛФ:",
            options=all_form_groups,
            default=all_form_groups,
        )

        only_verified = st.toggle(
            "Только подтвержденный лабораторный брак (Level 1–3)",
            value=True,
            help=(
                "Исключает превентивные отзывы производителей без указанного показателя брака "
                "в веб-реестре РЗН (Level 0) и письма о разрешении реализации."
            ),
        )

        st.divider()
        st.markdown("### 🔄 Синхронизация с API РЗН")
        if IS_CLOUD:
            # Публичная демо-версия: кнопка-заглушка, чтобы посетитель не мог изменить данные
            st.button(
                "Проверить новые письма API РЗН",
                width="stretch",
                disabled=True,
                help="В публичной демо-версии синхронизация отключена: данные обновляются в локальной среде разработчика.",
            )
            st.caption(
                "🔒 **Демо-версия.** Витрина работает на срезе данных, зафиксированном в репозитории. "
                "Живая синхронизация с API Росздравнадзора доступна только в локальной версии дашборда."
            )
        else:
            if st.button("Проверить новые письма API РЗН", width="stretch"):
                with st.spinner("Запрос к POST /ajax/services/lssearch..."):
                    try:
                        st.session_state["last_sync_result"] = trigger_incremental_update()
                    except Exception as exc:
                        st.session_state["last_sync_result"] = {
                            "last_sync_ts": datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
                            "latest_letter_date": "—",
                            "latest_letter_raw": f"ошибка синхронизации: {exc}",
                            "checked_records": "0",
                            "added_records": "0",
                            "status": "error",
                        }
                st.rerun()

            if "last_sync_result" in st.session_state:
                res_info = st.session_state["last_sync_result"]
                if res_info.get("status") == "ok":
                    st.success(
                        f"✅ **Синхронизация выполнена ({res_info['last_sync_ts']})**\n\n"
                        f"• Проверено записей: **{res_info['checked_records']}**\n"
                        f"• Новых записей добавлено: **{res_info['added_records']}**\n"
                        f"• Последнее письмо в базе: **{res_info['latest_letter_raw']}**"
                    )
                else:
                    st.warning(
                        f"⚠️ **Синхронизация не выполнена ({res_info['last_sync_ts']})**\n\n"
                        f"• {res_info['latest_letter_raw']}\n"
                        f"• Показан прежний срез данных — он не изменён."
                    )

        st.divider()
        st.caption(
            "Разработано химиком-аналитиком и инженером-технологом R&D (6+ лет в фармотрасли: ТЛФ, валидация, стабильность)"
        )

    # Применение фильтров к таблице фактов
    df_filtered = df_all[
        (df_all["is_recall_event"] == 1)
        & (df_all["year"] >= selected_years[0])
        & (df_all["year"] <= selected_years[1])
        & (df_all["dosage_form_group"].isin(selected_forms if selected_forms else all_form_groups))
    ].copy()

    if selected_origin != "Все площадки":
        df_filtered = df_filtered[df_filtered["origin_type"] == selected_origin]

    if only_verified:
        df_active = df_filtered[df_filtered["has_explicit_defect"] == 1].copy()
    else:
        df_active = df_filtered.copy()

    # Заголовок страницы
    st.title("💊 Мониторинг и скоринг технологического брака лекарственных средств в РФ")
    st.markdown(
        "Аналитическая система оценки качества производителей (**PRRI — Pharma Recall Risk Index**) "
        "на открытых данных государственного контроля **Росздравнадзора**."
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📊 1. Executive Summary & KPI",
            "🏭 2. Скоринг производителей (PRRI)",
            "🧪 3. Технологический R&D-анализ",
            "📐 4. Стат. гипотезы и методология PRRI",
        ]
    )

    # =========================================================================
    # ВКЛАДКА 1: EXECUTIVE SUMMARY & KPI
    # =========================================================================
    with tab1:
        verified_sub = df_filtered[df_filtered["has_explicit_defect"] == 1]
        total_batches = len(df_active)
        unique_letters_cnt = df_active["letter_number"].nunique()
        tech_crit_cnt = int((verified_sub["is_tech_or_critical_defect"] == 1).sum())
        tdr_val = round(tech_crit_cnt / max(1, len(verified_sub)) * 100.0, 1)
        total_prri_val = round(float(verified_sub["prri_incident_score"].sum()), 1)
        critical_mfg_cnt = int((df_mfg_rank["risk_tier"].str.startswith("Critical")).sum())
        spi_val = round(total_batches / max(1, unique_letters_cnt), 2)

        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            render_kpi_card(
                "Забракованных серий (в выборке)",
                f"{total_batches:,}".replace(",", " "),
                f"По {unique_letters_cnt} информ. письмам РЗН",
            )
        with k2:
            render_kpi_card(
                "Доля тех. и крит. брака (TDR)",
                f"{tdr_val}%",
                f"{tech_crit_cnt} серий (Level 2 формы + Level 3 АФС)",
            )
        with k3:
            render_kpi_card(
                "Суммарный индекс риска PRRI",
                f"{total_prri_val:,.1f}".replace(",", " "),
                "Взвешенный скор с учетом давности",
            )
        with k4:
            render_kpi_card(
                "Глубина поражения (Series / Letter)",
                f"{spi_val}",
                "Среднее число серий в одном письме",
            )
        with k5:
            render_kpi_card(
                "Площадок Critical Risk",
                f"{critical_mfg_cnt}",
                f"Из {len(df_mfg_rank)} производителей в базе",
            )

        st.markdown("<div style='margin-top: 18px;'></div>", unsafe_allow_html=True)
        st.markdown("#### 📈 Помесячная динамика забраковок и скользящее среднее `MA(3)` (SQL Window View)")
        df_m_sub = df_monthly[
            (df_monthly["year"] >= selected_years[0]) & (df_monthly["year"] <= selected_years[1])
        ].copy()

        fig_dyn = go.Figure()
        fig_dyn.add_trace(
            go.Bar(
                x=df_m_sub["year_month"],
                y=df_m_sub["level3_critical"],
                name="Level 3: Критический брак АФС и микробиологии (вес 5.0)",
                marker_color="#d62728",
            )
        )
        fig_dyn.add_trace(
            go.Bar(
                x=df_m_sub["year_month"],
                y=df_m_sub["level2_technological"],
                name="Level 2: Технологический брак формы (вес 2.5)",
                marker_color="#ff7f0e",
            )
        )
        fig_dyn.add_trace(
            go.Bar(
                x=df_m_sub["year_month"],
                y=df_m_sub["level1_packaging"],
                name="Level 1: Маркировка и упаковка (вес 1.0)",
                marker_color="#1f77b4",
            )
        )
        fig_dyn.add_trace(
            go.Scatter(
                x=df_m_sub["year_month"],
                y=df_m_sub["ma3_verified_defects"],
                mode="lines+markers",
                name="MA(3) — 3-мес. скользящее среднее (SQL AVG OVER)",
                line=dict(color="#111111", width=2.5),
            )
        )
        fig_dyn.update_layout(
            barmode="stack",
            height=410,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            xaxis_title="Месяц публикации письма РЗН",
            yaxis_title="Количество серий ЛС",
        )
        st.plotly_chart(fig_dyn, width="stretch")

        st.divider()
        st.markdown("#### 🧪 Структура брака по фармакопейно-технологическим категориям")
        cat_agg = (
            verified_sub.groupby(["defect_level", "defect_category"], as_index=False)
            .agg(batches=("batch_number", "count"), prri=("prri_incident_score", "sum"))
            .sort_values("batches", ascending=True)
        )
        color_map = {
            "Level 3: Критический брак АФС и микробиологии (5.0)": "#d62728",
            "Level 2: Технологический брак формы (2.5)": "#ff7f0e",
            "Level 1: Логистика и упаковка (1.0)": "#1f77b4",
        }
        max_batches_cat = int(cat_agg["batches"].max()) if not cat_agg.empty else 100
        fig_cat = px.bar(
            cat_agg,
            x="batches",
            y="defect_category",
            color="defect_level",
            orientation="h",
            text="batches",
            color_discrete_map=color_map,
            labels={
                "batches": "Забракованных серий",
                "defect_category": "Категория несоответствия",
                "defect_level": "Уровень критичности",
            },
        )
        fig_cat.update_traces(textposition="outside", cliponaxis=False)
        fig_cat.update_layout(
            height=430,
            xaxis=dict(range=[0, max_batches_cat * 1.16]),
            margin=dict(l=10, r=60, t=35, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        st.plotly_chart(fig_cat, width="stretch")

        st.divider()
        st.markdown("#### 💊 Поиск и рейтинг препаратов по индексу технологического риска PRRI")
        st.caption(
            "Вы можете найти любой интересующий препарат или производителя через поиск ниже, "
            "а также расширить таблицу до любого количества позиций."
        )

        all_drugs_agg = (
            verified_sub.groupby(
                ["trade_name", "manufacturer_clean", "origin_type", "dosage_form_short", "defect_category", "defect_level"],
                as_index=False,
            )
            .agg(
                Серий=("batch_number", "count"),
                Писем=("letter_number", "nunique"),
                PRRI=("prri_incident_score", lambda s: round(float(s.sum()), 2)),
                Последнее_письмо=("letter_date", "max"),
            )
            .sort_values(["PRRI", "Серий"], ascending=False)
            .reset_index(drop=True)
        )
        all_drugs_agg.insert(0, "Ранг", np.arange(1, len(all_drugs_agg) + 1))

        f_col1, f_col2 = st.columns([2.2, 1.0])
        with f_col1:
            drug_search_query = st.text_input(
                "🔎 Поиск препарата (по торговому наименованию, производителю или показателю брака):",
                value="",
                placeholder="Например: Эналаприл, Цитрамон, Ибупрофен, Спиронолактон, Велфарм...",
            )
        with f_col2:
            max_drugs_total = max(10, len(all_drugs_agg))
            top_n_drugs = st.selectbox(
                "Количество отображаемых позиций:",
                options=[15, 30, 50, 100, 250, max_drugs_total],
                index=2,
                format_func=lambda x: f"Все ({x})" if x == max_drugs_total else f"Топ-{x} позиций",
            )

        filtered_drugs_df = all_drugs_agg.copy()
        if drug_search_query.strip():
            q = drug_search_query.strip().lower()
            mask = (
                filtered_drugs_df["trade_name"].astype(str).str.lower().str.contains(q, na=False)
                | filtered_drugs_df["manufacturer_clean"].astype(str).str.lower().str.contains(q, na=False)
                | filtered_drugs_df["defect_category"].astype(str).str.lower().str.contains(q, na=False)
            )
            filtered_drugs_df = filtered_drugs_df[mask]

        st.dataframe(
            filtered_drugs_df.head(int(top_n_drugs)).rename(
                columns={
                    "trade_name": "Торговое наименование",
                    "manufacturer_clean": "Производитель",
                    "origin_type": "Локализация",
                    "dosage_form_short": "Лекарственная форма",
                    "defect_category": "Основной показатель брака",
                    "defect_level": "Уровень критичности",
                    "Последнее_письмо": "Последнее письмо РЗН",
                }
            ),
            width="stretch",
            hide_index=True,
            height=440,
        )

    # =========================================================================
    # ВКЛАДКА 2: РЕЙТИНГ И СКОРИНГ ПРОИЗВОДИТЕЛЕЙ (PRRI)
    # =========================================================================
    with tab2:
        st.markdown("#### 🎯 Наглядное сравнение заводов по индексу риска `PRRI` и доле тех. брака `TDR (%)`")
        st.info(
            "💡 **Почему важны обе метрики (`PRRI` и `TDR %`):** "
            "Например, у компании **«Юнифарм Инк»** отозвано 135 серий, но **TDR = 0%** (все 100% проблем — опечатки маркировки и упаковки). "
            "А у заводов из группы **Critical Risk** (`ООО \"ПРАНАФАРМ\"`, `ПАО \"Биосинтез\"`, `АО \"ВЕРОФАРМ\"`, `ООО \"Озон\"`) "
            "**TDR > 95%** — то есть почти каждая забраковка связана с химическим распадом субстанции или нерастворением таблетки."
        )

        ctrl1, ctrl2 = st.columns([1.4, 1.0])
        with ctrl1:
            chart_mode = st.radio(
                "Формат визуализации лидеров антирейтинга:",
                options=[
                    "📊 Рейтинг Топ-N заводов по PRRI (с указанием TDR %)",
                    "🧭 Карта рисков: Количество тех. серий vs Индекс PRRI (с подписями заводов)",
                ],
                horizontal=True,
            )
        with ctrl2:
            top_n_mfg_chart = st.slider(
                "Число заводов на графике:",
                min_value=10,
                max_value=40,
                value=20,
                step=5,
            )

        tier_colors = {
            "Critical Risk (Критический тех. риск)": "#d62728",
            "High Risk (Системный тех. брак)": "#ff7f0e",
            "Moderate Risk (Умеренный тех. риск)": "#f1c40f",
            "Packaging Risk (Преимущественно упаковка/маркировка)": "#1f77b4",
            "Low Risk (Низкий / единичный риск)": "#2ca02c",
        }

        mfg_top_df = df_mfg_rank.head(top_n_mfg_chart).copy()

        if chart_mode.startswith("📊"):
            mfg_bar_df = mfg_top_df.sort_values("prri_score", ascending=True).copy()
            mfg_bar_df["label_text"] = mfg_bar_df.apply(
                lambda r: f"PRRI: {r['prri_score']} | TDR: {r['tdr_pct']}% ({int(r['explicit_defects_count'])} сер.)",
                axis=1,
            )
            max_prri_val = float(mfg_bar_df["prri_score"].max()) if not mfg_bar_df.empty else 100.0
            fig_mfg = px.bar(
                mfg_bar_df,
                x="prri_score",
                y="manufacturer_clean",
                color="risk_tier",
                orientation="h",
                text="label_text",
                color_discrete_map=tier_colors,
                hover_data={
                    "origin_type": True,
                    "tdr_pct": True,
                    "explicit_defects_count": True,
                    "series_per_incident": True,
                    "primary_defect_category": True,
                    "label_text": False,
                },
                labels={
                    "prri_score": "Индекс технологического риска PRRI",
                    "manufacturer_clean": "Производитель",
                    "risk_tier": "Категория риска (Risk Tier)",
                },
            )
            fig_mfg.update_traces(textposition="outside", cliponaxis=False)
            fig_mfg.update_layout(
                height=max(460, top_n_mfg_chart * 26),
                xaxis=dict(range=[0, max_prri_val * 1.28]),
                margin=dict(l=20, r=80, t=30, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )
            st.plotly_chart(fig_mfg, width="stretch")
        else:
            # Наглядная карта без сжатия в 100%: ось X — число серий с браком (лог. шкала), ось Y — PRRI (лог. шкала)
            mfg_map_df = mfg_top_df.copy()
            mfg_map_df["short_name"] = mfg_map_df["manufacturer_clean"].str.replace(
                "Не установлен (фальсификат / без маркировки)", "Фальсификат без маркировки", regex=False
            )
            fig_scatter = px.scatter(
                mfg_map_df,
                x="explicit_defects_count",
                y="prri_score",
                size="series_per_incident",
                color="risk_tier",
                text="short_name",
                hover_name="manufacturer_clean",
                hover_data={
                    "origin_type": True,
                    "prri_score": True,
                    "tdr_pct": True,
                    "explicit_defects_count": True,
                    "series_per_incident": True,
                    "primary_defect_category": True,
                    "short_name": False,
                },
                log_x=True,
                log_y=True,
                size_max=34,
                color_discrete_map=tier_colors,
                labels={
                    "explicit_defects_count": "Количество забракованных серий Level 1–3 (лог. шкала)",
                    "prri_score": "Индекс технологического риска PRRI (лог. шкала)",
                    "risk_tier": "Категория риска (Risk Tier)",
                    "series_per_incident": "Серий на 1 письмо",
                },
            )
            fig_scatter.update_traces(
                textposition="top center",
                textfont=dict(size=10),
                marker=dict(line=dict(width=1, color="#333333"), opacity=0.88),
            )
            fig_scatter.update_layout(
                height=520,
                margin=dict(l=20, r=40, t=40, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )
            st.plotly_chart(fig_scatter, width="stretch")

        st.markdown("#### 🏆 Полный рейтинг производителей (SQL-витрина `v_manufacturer_risk_ranking` с кривой Парето)")
        st.dataframe(
            df_mfg_rank[
                [
                    "overall_risk_rank",
                    "manufacturer_clean",
                    "origin_type",
                    "risk_tier",
                    "prri_score",
                    "market_risk_share_pct",
                    "cumulative_pareto_risk_pct",
                    "tdr_pct",
                    "explicit_defects_count",
                    "series_per_incident",
                    "primary_defect_category",
                    "last_recall_date",
                ]
            ].rename(
                columns={
                    "overall_risk_rank": "Ранг",
                    "manufacturer_clean": "Производитель",
                    "origin_type": "Локализация",
                    "risk_tier": "Risk Tier",
                    "prri_score": "PRRI Score",
                    "market_risk_share_pct": "Доля риска рынка (%)",
                    "cumulative_pareto_risk_pct": "Кумулятивный Парето (%)",
                    "tdr_pct": "TDR (%)",
                    "explicit_defects_count": "Серий (Level 1–3)",
                    "series_per_incident": "Серий / письмо",
                    "primary_defect_category": "Главная уязвимость завода",
                    "last_recall_date": "Последнее письмо",
                }
            ),
            width="stretch",
            hide_index=True,
            height=360,
        )

        st.divider()
        st.markdown("### 🔎 Интерактивное досье поставщика (Supplier Risk Card & Radar Chart)")
        mfg_list = df_mfg_rank["manufacturer_clean"].tolist()
        selected_mfg = st.selectbox(
            "Выберите или введите название фармпроизводителя для детального аудита:",
            options=mfg_list,
            index=0,
        )

        mfg_row = df_mfg_rank[df_mfg_rank["manufacturer_clean"] == selected_mfg].iloc[0]
        mfg_incidents = df_all[
            (df_all["manufacturer_clean"] == selected_mfg) & (df_all["is_recall_event"] == 1)
        ].sort_values("letter_date", ascending=False)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            render_kpi_card(
                "Индекс риска PRRI",
                f"{mfg_row['prri_score']}",
                f"Ранг #{int(mfg_row['overall_risk_rank'])} из {len(df_mfg_rank)} на рынке РФ",
            )
        with c2:
            render_kpi_card(
                "Доля тех. брака (TDR)",
                f"{mfg_row['tdr_pct']}%",
                f"Категория: {mfg_row['risk_tier']}",
            )
        with c3:
            render_kpi_card(
                "Забраковано серий / писем РЗН",
                f"{int(mfg_row['explicit_defects_count'])} / {int(mfg_row['unique_letters'])}",
                f"Средняя глубина: {mfg_row['series_per_incident']} серий на письмо",
            )
        with c4:
            render_kpi_card(
                "Главная технологическая проблема завода",
                f"{int(mfg_row['primary_defect_batches'])} сер.",
                f"{mfg_row['primary_defect_category']}",
            )

        st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)
        r_left, r_right = st.columns([1.0, 1.25], gap="large")
        with r_left:
            radar_cats = [
                "Родственные и посторонние примеси (стабильность АФС)",
                "Количественное определение и содержание АФС",
                "Растворение и распадаемость",
                "Описание, цветность и прозрачность",
                "Однородность дозирования и массы",
                "Механические включения",
                "Микробиология, стерильность и эндотоксины",
                "Маркировка и упаковка",
            ]
            mfg_ver = mfg_incidents[mfg_incidents["has_explicit_defect"] == 1]
            cat_counts = mfg_ver["defect_category"].value_counts()
            r_values = [int(cat_counts.get(c, 0)) for c in radar_cats]
            short_labels = [
                "Примеси (АФС)",
                "Колич. опред.",
                "Растворение",
                "Описание / Цвет",
                "Однородность",
                "Мех. включения",
                "Микробиология",
                "Маркировка/Упак.",
            ]

            fig_radar = go.Figure()
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=r_values + [r_values[0]],
                    theta=short_labels + [short_labels[0]],
                    fill="toself",
                    name=selected_mfg,
                    line_color="#d62728",
                )
            )
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, rangemode="tozero")),
                title=dict(text=f"Профиль дефектов: {selected_mfg}", font=dict(size=14)),
                height=400,
                margin=dict(l=75, r=75, t=55, b=45),
            )
            st.plotly_chart(fig_radar, width="stretch")

        with r_right:
            st.markdown(f"**Реестр писем Росздравнадзора по площадке `{selected_mfg}` (с прямыми ссылками на PDF):**")
            st.dataframe(
                mfg_incidents[
                    [
                        "letter_date",
                        "letter_number",
                        "trade_name",
                        "batch_number",
                        "defect_parameter",
                        "regulatory_action",
                        "prri_incident_score",
                        "letter_link",
                    ]
                ].rename(
                    columns={
                        "letter_date": "Дата письма",
                        "letter_number": "№ письма",
                        "trade_name": "Препарат",
                        "batch_number": "Серия",
                        "defect_parameter": "Показатель брака (col11)",
                        "regulatory_action": "Мера РЗН",
                        "prri_incident_score": "PRRI серии",
                        "letter_link": "PDF письма РЗН",
                    }
                ),
                column_config={
                    "PDF письма РЗН": st.column_config.LinkColumn(
                        "PDF письма РЗН",
                        display_text="Скачать PDF 📥",
                    )
                },
                width="stretch",
                hide_index=True,
                height=380,
            )

    # =========================================================================
    # ВКЛАДКА 3: ТЕХНОЛОГИЧЕСКИЙ R&D-АНАЛИЗ
    # =========================================================================
    with tab3:
        st.markdown("#### 🧬 Тепловая матрица R&D: `Группа лекарственной формы × Категория брака` (`v_dosage_form_defect_matrix`)")
        pivot_pct = df_form_matrix.pivot(
            index="dosage_form_group",
            columns="defect_category",
            values="share_within_form_pct",
        ).fillna(0.0)

        fig_heat = px.imshow(
            pivot_pct,
            text_auto=".1f",
            aspect="auto",
            color_continuous_scale="YlOrRd",
            labels=dict(
                x="Фармакопейно-технологическая категория дефекта",
                y="Группа лекарственных форм",
                color="Доля внутри ЛФ (%)",
            ),
        )
        fig_heat.update_layout(height=420, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_heat, width="stretch")

        st.divider()
        st.markdown("#### 🔁 Витрина хронических серийных рецидивов по препаратам (`v_chronic_drug_recidivism`)")
        st.caption(
            "Рассчитано в SQL через оконные функции `LAG(letter_date) OVER (PARTITION BY manufacturer, drug, defect)` "
            "и `DENSE_RANK()`. Позволяет отличить разовый отзыв партии от многолетней неотвалидированной технологии."
        )

        ch_col1, ch_col2, ch_col3 = st.columns([1.6, 1.0, 1.0])
        with ch_col1:
            chronic_search = st.text_input(
                "🔎 Фильтр по препарату, производителю или дефекту:",
                value="",
                placeholder="Например: Эналаприл, Растворение, Примеси...",
                key="chronic_search_input",
            )
        with ch_col2:
            min_prri_chronic = st.slider(
                "Минимальный порог PRRI кейса:",
                min_value=0.0,
                max_value=200.0,
                value=0.0,
                step=5.0,
            )
        with ch_col3:
            max_chronic_rows = len(df_chronic)
            chronic_limit = st.selectbox(
                "Показывать строк в витрине:",
                options=[50, 100, 200, max_chronic_rows],
                index=1,
                format_func=lambda x: f"Все ({x} кейсов)" if x == max_chronic_rows else f"Топ-{x} кейсов",
            )

        df_chronic_view = df_chronic[df_chronic["drug_defect_prri_score"] >= min_prri_chronic].copy()
        if chronic_search.strip():
            cq = chronic_search.strip().lower()
            cmask = (
                df_chronic_view["trade_name"].astype(str).str.lower().str.contains(cq, na=False)
                | df_chronic_view["manufacturer_clean"].astype(str).str.lower().str.contains(cq, na=False)
                | df_chronic_view["defect_category"].astype(str).str.lower().str.contains(cq, na=False)
            )
            df_chronic_view = df_chronic_view[cmask]

        st.dataframe(
            df_chronic_view.head(int(chronic_limit))[
                [
                    "chronic_case_rank",
                    "manufacturer_clean",
                    "trade_name",
                    "dosage_form_short",
                    "defect_category",
                    "unique_letters_count",
                    "total_recalled_batches",
                    "series_per_letter_ratio",
                    "first_incident_date",
                    "last_incident_date",
                    "drug_defect_prri_score",
                ]
            ].rename(
                columns={
                    "chronic_case_rank": "Ранг",
                    "manufacturer_clean": "Производитель",
                    "trade_name": "Препарат",
                    "dosage_form_short": "Лекарственная форма",
                    "defect_category": "Категория технологического дефекта",
                    "unique_letters_count": "Писем РЗН",
                    "total_recalled_batches": "Всего серий",
                    "series_per_letter_ratio": "Серий / письмо",
                    "first_incident_date": "Первый инцидент",
                    "last_incident_date": "Последний инцидент",
                    "drug_defect_prri_score": "PRRI кейса",
                }
            ),
            width="stretch",
            hide_index=True,
            height=520,
        )

        st.divider()
        st.markdown("#### 🔬 Экспертный разбор реальных R&D-кейсов из реестра Росздравнадзора")
        st.info(
            "**📌 Откуда взяты эти кейсы и зачем они нужны бизнесу?**\n\n"
            "1. **Источник фактов:** Все цифры ниже (конкретные препараты, заводы, количество отозванных серий и точные формулировки брака из поля `col11`) "
            "**напрямую выгружены из витрины рецидивов `v_chronic_drug_recidivism`**.\n"
            "2. **Источник интерпретации:** Объяснение физико-химических и технологических механизмов брака выполнено на базе **6-летнего опыта автора в R&D фармацевтики** "
            "(разработка ТЛФ, валидация, изучение стабильности и расследование OOS).\n"
            "3. **Практическая ценность для бизнеса:**\n"
            "   - **Для закупок (аптечные сети, дистрибьюторы, тендерные отделы):** Позволяет отличить случайный сбой партии от *хронической нестабильности рецептуры*. "
            "Если препарат годами отзывается по деградации субстанции (как Эналаприл или Цитрамон у конкретных заводов), новые поставки с высокой вероятностью снова попадут под отзыв РЗН через 6–12 месяцев хранения — закупщик несет прямые убытки на возврат и заморозку стока.\n"
            "   - **Для R&D и дирекции по качеству (QA/QC):** Дает готовый реестр критических точек контроля (CAPA) при разработке дженериков, смене поставщика АФС и выборе барьерной упаковки."
        )

        e1, e2 = st.columns(2)
        with e1:
            with st.expander("1️⃣ Кейс «Растворение» ТЛФ: Спиронолактон (32 серии), Ибупрофен Велфарм, Танацехол (14 серий)", expanded=True):
                st.markdown(
                    """
                    - **Факт из базы РЗН:** 547 забракованных серий ТЛФ по тесту «Растворение» и «Распадаемость» (`20.0%` всех дефектов таблеток и капсул). У препарата «Спиронолактон» (ООО «ПРАНАФАРМ») — 32 отозванные серии в 19 письмах.
                    - **Технологическая причина (R&D):**
                      - Для плохо растворимых в воде субстанций (BCS Class II — спиронолактон, ибупрофен) смена поставщика АФС с более крупным размером частиц ($d_{90}$) без адаптации количества дезинтегранта и режима грануляции резко замедляет растворение.
                      - Дополнительный фактор — перепрессовка таблетки или «сшивка» (cross-linking) желатина капсул при хранении.
                    - **Решение для бизнеса и R&D:** Входной контроль гранулометрии АФС (лазерная дифракция) и стресс-тест растворения в 3 средах pH перед закупкой новой серии субстанции.
                    """
                )
            with st.expander("2️⃣ Кейс «Родственные примеси»: Эналаприл (108 серий в 47 письмах) и Капотен (каптоприла дисульфид)", expanded=True):
                st.markdown(
                    """
                    - **Факт из базы РЗН:** у ПАО «Биосинтез» по препарату «Эналаприл» вышло **47 отдельных писем РЗН** (108 серий подряд за 2023–2024 гг.) по показателю «Родственные примеси»; у «Капотена» — отзыв по специфической примеси `Каптоприла дисульфид`.
                    - **Химико-технологическая причина:**
                      - Эналаприла малеат во влажной среде и при неоптимальном pH таблеточной массы распадается до **эналаприлата** и циклизуется в **дикетопиперазин**.
                      - Каптоприл окисляется по тиольной группе $-\\text{SH}$ кислородом воздуха до дисульфида.
                    - **Решение для бизнеса и R&D:** Жесткий контроль остаточной влаги гранулята (LOD) и перевод препарата из обычного ПВХ-блистера в высокобарьерный блистер **Ал/Ал (холодная формовка)** или ПВХ/ПВДХ.
                    """
                )
        with e2:
            with st.expander("3️⃣ Кейс гидролиза АСК: Цитрамон П (27 серий в 25 письмах — свободная салициловая кислота)", expanded=True):
                st.markdown(
                    """
                    - **Факт из базы РЗН:** регулярный отзыв серий «Цитрамона П» (ОАО «ДАЛЬХИМФАРМ», 25 писем за 2020–2026 гг.) по показателям `Описание; Родственные примеси (салициловая кислота)` и `Количественное определение`.
                    - **Химико-технологическая причина:**
                      - При избыточной влаге крахмала или негерметичной упаковке ацетилсалициловая кислота подвергается гидролизу с отщеплением уксусной кислоты и ростом кристаллов **салициловой кислоты** (одновременный провал по «Описанию» и «Примесям»).
                    - **Решение для бизнеса и R&D:** Для дистрибьюторов — не закупать длинный сток препарата у площадок с историей гидролиза АСК; для технологов — использование предварительно высушенных эксципиентов с низкой активностью воды ($a_w$).
                    """
                )
            with st.expander("4️⃣ Кейс парентеральных форм: Механические включения (153 серии инъекционных растворов)", expanded=True):
                st.markdown(
                    """
                    - **Факт из базы РЗН:** у производителей ампульных и инфузионных растворов (`ФКП «Армавирская биофабрика»` — 24 серии, `ОАО «Самарамедпром»` и др.) `Механические включения` и `Прозрачность` являются главной причиной отзывов.
                    - **Технологическая причина:**
                      - Проскок микрочастиц через мембранные фильтры ($0.22\\ \\mu\\text{m}$), микросколы стекла при запайке ампул или деламинация (отслоение чешуек кремнезема) стекла при автоклавировании.
                    - **Решение для бизнеса и R&D:** Переход на ампулы 1-го гидролитического класса (боросиликатное стекло) и 100% автоматическая оптическая инспекция ампул после стерилизации.
                    """
                )

    # =========================================================================
    # ВКЛАДКА 4: ПРОВЕРКА СТАТ. ГИПОТЕЗ И МЕТОДОЛОГИЯ РАСЧЕТА PRRI
    # =========================================================================
    with tab4:
        st.markdown("### 📐 Как устроен индекс риска PRRI (простыми словами)")
        st.markdown(
            "Если оценивать заводы просто по количеству отозванных серий, то компания с опечаткой на 100 пачках витаминов "
            "будет выглядеть «опаснее», чем завод, выпустивший 10 серий токсичного инъекционного препарата. "
            "Чтобы решить эту проблему, мы разработали индекс **PRRI (Pharma Recall Risk Index)**, который перемножает **4 понятных фактора** для каждой забракованной серии:"
        )
        st.latex(
            r"\text{PRRI} = \sum_{i=1}^{N} \left( W_{\text{defect}, i} \times W_{\text{action}, i} \times W_{\text{chronic}, i} \times e^{-\lambda \cdot \Delta t_i} \right)"
        )

        m_col1, m_col2 = st.columns([1.15, 0.85], gap="large")
        with m_col1:
            st.markdown(
                """
                1. **Тяжесть брака ($W_{\\text{defect}}$):**
                   - **Уровень 1 (вес `1.0`) — Только упаковка:** опечатка в маркировке или мятая пачка. Сама таблетка безопасна.
                   - **Уровень 2 (вес `2.5`) — Проблема с формой:** таблетка плохо растворяется, крошится или изменила цвет. Препарат может не подействовать вовремя.
                   - **Уровень 3 (вес `5.0`) — Угроза здоровью:** в лекарстве выросли токсичные примеси, нарушена дозировка действующего вещества или препарат нестерилен.
                2. **Строгость меры Росздравнадзора ($W_{\\text{action}}$):**
                   - Полный отзыв / изъятие из аптек = **`1.5`**; временная приостановка = **`1.0`**. *(Письма о возврате в оборот после проверки получают вес **`0.0`**).*
                3. **Штраф за системность ($W_{\\text{chronic}}$):**
                   - Если завод наступает на одни и те же технологические грабли снова и снова ($N > 1$ серий по одному дефекту), включается повышающий множитель $1 + 0.25 \\cdot \\ln(N)$.
                4. **Свежесть инцидента ($e^{-\\lambda \\cdot \\Delta t}$):**
                   - Забраковки за последние **12 месяцев** учитываются на **100%** (вес `1.0`), а старые ошибки 3–5-летней давности плавно теряют вес, чтобы рейтинг отражал текущее состояние завода.
                """
            )
        with m_col2:
            st.markdown("#### 🧮 Калькулятор-симулятор PRRI")
            st.caption("Подвигайте ползунки, чтобы увидеть, как система оценивает риск разных ситуаций:")
            sim_level = st.selectbox(
                "Какой брак обнаружен?",
                options=[
                    ("Уровень 3: Примеси / Дозировка / Стерильность (вес 5.0)", 5.0),
                    ("Уровень 2: Растворение / Описание / Однородность (вес 2.5)", 2.5),
                    ("Уровень 1: Ошибка маркировки или упаковки (вес 1.0)", 1.0),
                ],
                format_func=lambda x: x[0],
            )
            sim_action = st.radio(
                "Решение Росздравнадзора:",
                options=[("Отзыв / Изъятие (1.5)", 1.5), ("Приостановление (1.0)", 1.0)],
                format_func=lambda x: x[0],
                horizontal=True,
            )
            sim_batches = st.slider("Сколько серий с этой проблемой у завода (N):", 1, 50, 5)
            sim_age = st.slider("Сколько лет прошло с даты письма:", 0.0, 6.0, 0.5, 0.5)

            w_def_sim = sim_level[1]
            w_act_sim = sim_action[1]
            w_chr_sim = 1.0 + 0.25 * np.log(sim_batches) if (w_def_sim > 1.0 and sim_batches > 1) else 1.0
            decay_sim = float(np.exp(-0.50 * max(0.0, sim_age - 1.0)))
            single_score = w_def_sim * w_act_sim * w_chr_sim * decay_sim
            batch_total_score = single_score * sim_batches

            st.success(
                f"**Балл риска за 1 серию:** `{single_score:.2f}`  \n"
                f"*(Тяжесть `{w_def_sim}` × Мера `{w_act_sim}` × Системность `{w_chr_sim:.2f}` × Свежесть `{decay_sim:.2f}`)*  \n\n"
                f"👉 **Итоговый вклад всех {sim_batches} серий в PRRI завода:** **`{batch_total_score:.1f}`**"
            )

        st.divider()
        st.markdown("### 🧪 Проверка 4 статистических гипотез на данных 2020–2026 гг.")
        st.caption(
            "Мы проверили 4 ключевых предположения о рынке лекарств с помощью строгих математических тестов (`scipy.stats`). "
            "Ниже — краткая суть того, что мы проверили и какие инсайты получили:"
        )

        hyp = load_hypothesis_results()

        # Человекочитаемые короткие карточки для 4 гипотез
        friendly_cards = [
            {
                "key": "h1_origin_vs_defect_level",
                "question": "1️⃣ Чем отличается брак российских и импортных лекарств?",
                "verdict": "✅ ПОДТВЕРЖДЕНО: Причины брака кардинально различаются (p < 0.0001)",
                "insight": (
                    "**Какой инсайт мы нашли:** У импортных лекарств **35.2% всех проблем — это упаковка и маркировка** "
                    "(ошибки перевода пачки и комплектации при ввозе в РФ), а технологический брак самой формы составляет лишь **14.6%**. "
                    "У российских заводов ситуация обратная: проблемы с упаковкой возникают вдвое реже (**15.9%**), "
                    "зато **38.6% забраковок — это технологический брак самой таблетки или ампулы** (не растворяется, изменила цвет, содержит видимые частицы)."
                ),
            },
            {
                "key": "h2_dosage_form_vs_tdr",
                "question": "2️⃣ Какие лекарственные формы сложнее всего производить без тех. брака?",
                "verdict": "✅ ПОДТВЕРЖДЕНО: Таблетки и капсулы страдают от тех. брака сильнее жидких форм (p < 0.0001)",
                "insight": (
                    "**Какой инсайт мы нашли:** В **твердых формах (таблетки и капсулы)** доля реального технологического и химического брака достигает **83.9%** "
                    "(главные уязвимости — рост примесей и плохое растворение). А в **нестерильных жидкостях (сиропы, капли, растворы)** "
                    "почти половина всех отзывов (**42.9%**) связана просто с этикеткой и пачкой."
                ),
            },
            {
                "key": "h3_prri_mann_whitney",
                "question": "3️⃣ У кого выше итоговый индекс технологического риска PRRI — у заводов РФ или Импорта?",
                "verdict": "✅ ПОДТВЕРЖДЕНО: У российских площадок медианный риск PRRI в 2.7 раза выше (p = 0.0011)",
                "insight": (
                    "**Какой инсайт мы нашли:** Медианный индекс риска российских заводов составляет **4.37** против **1.64** у зарубежных площадок. "
                    "Главная причина разрыва — **серийная повторяемость брака**: российские заводы из нижней части рейтинга часто допускают повторные отзывы десятков серий одного и того же препарата на протяжении 1–3 лет."
                ),
            },
            {
                "key": "h4_seasonality_kruskal",
                "question": "4️⃣ Правда ли, что летом или зимой Росздравнадзор бракует больше лекарств (сезонность)?",
                "verdict": "⚪ МИФ ОПРОВЕРГНУТ: Календарной сезонности в письмах РЗН нет (p = 0.5269)",
                "insight": (
                    "**Какой инсайт мы нашли:** Количество забракованных серий статистически не зависит от времени года (медиана держится на уровне **27–35 серий в месяц** в любом квартале). "
                    "**Почему так происходит:** Между моментом, когда лекарство перегрели при перевозке летом, и выходом официального письма РЗН проходит **от 3 до 9 месяцев** "
                    "(пока препарат отберут в аптеке, доставят в федеральную лабораторию и проведут анализы). Эта бюрократическая и лабораторная задержка полностью стирает сезонные пики."
                ),
            },
        ]

        for card in friendly_cards:
            item = hyp[card["key"]]
            with st.container(border=True):
                st.markdown(f"#### {card['question']}")
                st.markdown(f"**Статистический вердикт:** `{card['verdict']}` | *({item['test_name']}: {item['stat_name']} = {item['stat_value']}, {item['effect_size_name']} = {item['effect_size_value']})*")
                st.markdown(card["insight"])

                with st.expander("📊 Показать таблицу с цифрами и формулировку H0 / H1"):
                    st.markdown(f"- **Нулевая гипотеза ($H_0$):** {item['h0']}\n- **Альтернативная гипотеза ($H_1$):** {item['h1']}")
                    if "contingency_pct" in item:
                        st.caption("Распределение в процентах по строкам (%):")
                        st.dataframe(item["contingency_pct"], width="stretch")
                    if "summary_table" in item:
                        st.dataframe(item["summary_table"], width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
