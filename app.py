"""
Интерактивный аналитический веб-дашборд (Streamlit + Plotly):
«Мониторинг и скоринг технологического брака, отзывов и стабильности ЛС на рынке РФ (PRRI)»
На живых данных Росздравнадзора (2020–2026).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.cleaner import build_cleaned_dataset
from src.database import DEFAULT_SQLITE_PATH, get_db_engine, init_database, query_df
from src.scraper import RoszdravnadzorScraper
from src.stats_tests import run_all_hypothesis_tests

st.set_page_config(
    page_title="PRRI — Мониторинг и скоринг брака ЛС (Росздравнадзор)",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Кастомная стилизация карточек и бейджей
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2rem;
    }
    div[data-testid="stMetric"] {
        background-color: rgba(28, 131, 225, 0.06);
        border: 1px solid rgba(28, 131, 225, 0.2);
        padding: 12px 16px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def ensure_data_ready() -> None:
    """Проверяет наличие подготовленной БД; при первом старте контейнера собирает витрины."""
    raw_path = Path("data/raw/rzn_full_raw.csv")
    cleaned_path = Path("data/processed/rzn_cleaned.csv")
    mfg_path = Path("data/processed/manufacturers_prri.csv")

    if not raw_path.exists():
        sample_path = Path("data/raw/sample_100.csv")
        if sample_path.exists() and not cleaned_path.exists():
            with st.spinner("Первичная выгрузка реестра Росздравнадзора (2020–2026)..."):
                scraper = RoszdravnadzorScraper()
                parsed_rows, _ = scraper.fetch_records(date_from="01.01.2020", page_size=500, max_records=None)
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(parsed_rows).to_csv(raw_path, index=False, encoding="utf-8-sig")

    if raw_path.exists() and (not cleaned_path.exists() or not mfg_path.exists()):
        with st.spinner("Очистка данных и расчет индекса PRRI..."):
            build_cleaned_dataset(
                raw_csv_path=raw_path,
                cleaned_csv_path=cleaned_path,
                mfg_csv_path=mfg_path,
            )

    engine = get_db_engine()
    try:
        query_df("SELECT 1 FROM v_manufacturer_risk_ranking LIMIT 1", engine=engine)
    except Exception:
        if cleaned_path.exists() and mfg_path.exists():
            with st.spinner("Инициализация реляционной БД и SQL-витрин..."):
                init_database(cleaned_csv_path=cleaned_path, mfg_csv_path=mfg_path)


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


def trigger_incremental_update() -> tuple[int, str]:
    """Догружает свежие записи за последние 30 дней из API Росздравнадзора и пересчитывает БД."""
    raw_path = Path("data/raw/rzn_full_raw.csv")
    scraper = RoszdravnadzorScraper()
    fresh_rows, _ = scraper.fetch_records(date_from="01.01.2026", page_size=200, max_records=200)
    df_fresh = pd.DataFrame(fresh_rows)

    if raw_path.exists():
        df_old = pd.read_csv(raw_path)
        before_cnt = len(df_old)
        df_combined = pd.concat([df_fresh, df_old], ignore_index=True)
        df_combined = df_combined.drop_duplicates(
            subset=["letter_raw", "trade_name", "batch_number", "manufacturer_raw", "defect_parameter"]
        )
        added = max(0, len(df_combined) - before_cnt)
        df_combined.to_csv(raw_path, index=False, encoding="utf-8-sig")
    else:
        df_fresh.to_csv(raw_path, index=False, encoding="utf-8-sig")
        added = len(df_fresh)

    build_cleaned_dataset()
    init_database()
    st.cache_data.clear()
    max_letter = df_fresh["letter_raw"].iloc[0] if not df_fresh.empty else "актуально"
    return added, max_letter


def main() -> None:
    data = load_dashboard_tables()
    df_all = data["fact_recalls"]
    df_monthly = data["v_monthly"]
    df_mfg_rank = data["v_mfg_rank"]
    df_form_matrix = data["v_form_matrix"]
    df_chronic = data["v_chronic"]
    db_dialect = data["db_dialect"]["dialect"].iloc[0]

    # -------------------------------------------------------------------------
    # SIDEBAR: Панель управления, фильтры и синхронизация с API Росздравнадзора
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.markdown("## 💊 PRRI Analytics")
        st.caption(
            "Мониторинг и скоринг технологического брака, отзывов и стабильности ЛС на рынке РФ"
        )
        st.markdown(
            f"**СУБД:** `{db_dialect}` | **Источник:** `roszdravnadzor.gov.ru`  \n"
            f"**Актуальность среза:** `{df_all['letter_date'].max()}`"
        )
        st.divider()

        st.markdown("### 🔍 Глобальные фильтры")
        min_year = int(df_all["year"].min())
        max_year = int(df_all["year"].max())
        selected_years = st.slider(
            "Период писем Росздравнадзора (годы):",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year),
        )

        origin_options = ["Все площадки", "Отечественный (РФ)", "Зарубежный (Импорт)"]
        selected_origin = st.selectbox("Локализация производства:", origin_options, index=0)

        all_form_groups = sorted(df_all["dosage_form_group"].dropna().unique().tolist())
        selected_forms = st.multiselect(
            "Технологические группы ЛФ:",
            options=all_form_groups,
            default=all_form_groups,
        )

        only_verified = st.toggle(
            "Только верифицированные лабораторные дефекты (Level 1–3)",
            value=True,
            help=(
                "Исключает превентивные отзывы производителей без указанного показателя брака "
                "в веб-реестре РЗН (Level 0) и письма о разрешении реализации."
            ),
        )

        st.divider()
        st.markdown("### 🔄 Живая синхронизация с РЗН")
        if st.button("Проверить новые письма API РЗН", width="stretch"):
            with st.spinner("Запрос к POST /ajax/services/lssearch..."):
                added_cnt, latest_letter = trigger_incremental_update()
            st.success(f"Синхронизировано! Новых записей: {added_cnt}. Последнее письмо: {latest_letter}")
            st.rerun()

        st.divider()
        st.caption(
            "Разработано химиком-аналитиком и технологом R&D (6+ лет в фармотрасли: ТЛФ, валидация, стабильность)"
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
        k1.metric(
            "Забракованных серий (в выборке)",
            f"{total_batches:,}".replace(",", " "),
            f"{unique_letters_cnt} писем РЗН",
        )
        k2.metric(
            "Доля тех. и крит. брака (TDR)",
            f"{tdr_val}%",
            f"{tech_crit_cnt} серий Level 2 + Level 3",
        )
        k3.metric(
            "Суммарный индекс риска PRRI",
            f"{total_prri_val:,.1f}".replace(",", " "),
            "С учетом Recency Decay",
        )
        k4.metric(
            "Глубина поражения (Series/Letter)",
            f"{spi_val}",
            "Среднее число серий на письмо",
        )
        k5.metric(
            "Площадок Critical Risk",
            f"{critical_mfg_cnt}",
            f"из {len(df_mfg_rank)} производителей",
        )

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
            height=420,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            xaxis_title="Месяц публикации письма РЗН",
            yaxis_title="Количество серий ЛС",
        )
        st.plotly_chart(fig_dyn, width="stretch")

        col_left, col_right = st.columns([1.15, 0.85])
        with col_left:
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
            fig_cat.update_traces(textposition="outside")
            fig_cat.update_layout(
                height=430,
                margin=dict(l=10, r=30, t=20, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )
            st.plotly_chart(fig_cat, width="stretch")

        with col_right:
            st.markdown("#### 💊 Топ-12 препаратов по индексу технологического риска PRRI")
            top_drugs = (
                verified_sub.groupby(["trade_name", "manufacturer_clean", "defect_category"], as_index=False)
                .agg(
                    Серий=("batch_number", "count"),
                    Писем=("letter_number", "nunique"),
                    PRRI=("prri_incident_score", lambda s: round(float(s.sum()), 1)),
                )
                .sort_values("PRRI", ascending=False)
                .head(12)
            )
            st.dataframe(
                top_drugs.rename(
                    columns={
                        "trade_name": "Торговое наименование",
                        "manufacturer_clean": "Производитель",
                        "defect_category": "Основной показатель брака",
                    }
                ),
                width="stretch",
                hide_index=True,
                height=400,
            )

    # =========================================================================
    # ВКЛАДКА 2: РЕЙТИНГ И СКОРИНГ ПРОИЗВОДИТЕЛЕЙ (PRRI)
    # =========================================================================
    with tab2:
        st.markdown("#### 🎯 Матрица риска поставщиков: `PRRI Score × Доля тех. брака TDR (%)`")
        st.info(
            "💡 **Продуктовый кейс валидации метрики:** Обратите внимание на точку **«Юнифарм Инк»** внизу справа "
            "(135 отозванных серий, но **TDR = 0.0%** — 100% инцидентов связаны только с маркировкой и упаковкой). "
            "В обычном количественном отчете такой поставщик попал бы в черный список, тогда как наша матрица "
            "`PRRI × TDR` автоматически отделяет упаковочный сбой (`Packaging Risk`) от заводов с критическим "
            "распадом АФС и провалом растворения (`ООО \"ПРАНАФАРМ\"`, `ПАО \"Биосинтез\"`, `АО \"ВЕРОФАРМ\"`, `ООО \"Озон\"`)."
        )

        mfg_plot_df = df_mfg_rank[df_mfg_rank["explicit_defects_count"] >= 3].copy()
        tier_colors = {
            "Critical Risk (Критический тех. риск)": "#d62728",
            "High Risk (Системный тех. брак)": "#ff7f0e",
            "Moderate Risk (Умеренный тех. риск)": "#f1c40f",
            "Packaging Risk (Преимущественно упаковка/маркировка)": "#1f77b4",
            "Low Risk (Низкий / единичный риск)": "#2ca02c",
        }
        fig_scatter = px.scatter(
            mfg_plot_df,
            x="tdr_pct",
            y="prri_score",
            size="explicit_defects_count",
            color="risk_tier",
            hover_name="manufacturer_clean",
            hover_data={
                "origin_type": True,
                "prri_score": True,
                "tdr_pct": True,
                "explicit_defects_count": True,
                "series_per_incident": True,
                "primary_defect_category": True,
            },
            log_y=True,
            size_max=42,
            color_discrete_map=tier_colors,
            labels={
                "tdr_pct": "TDR (%) — Доля технологического и критического брака (Level 2 + Level 3)",
                "prri_score": "Индекс риска PRRI (лог. шкала)",
                "risk_tier": "Категория риска (Risk Tier)",
                "explicit_defects_count": "Верифицированных серий",
            },
        )
        fig_scatter.add_vline(x=40.0, line_dash="dash", line_color="gray", annotation_text="Граница Packaging / Tech Risk (TDR=40%)")
        fig_scatter.add_hline(y=150.0, line_dash="dot", line_color="#d62728", annotation_text="Порог Critical Risk (PRRI=150)")
        fig_scatter.update_layout(height=450, margin=dict(l=20, r=20, t=30, b=20))
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
            height=340,
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
        c1.metric("Индекс риска PRRI", f"{mfg_row['prri_score']}", f"Ранг #{int(mfg_row['overall_risk_rank'])} на рынке")
        c2.metric("Доля тех. брака (TDR)", f"{mfg_row['tdr_pct']}%", mfg_row["risk_tier"].split("(")[0].strip())
        c3.metric(
            "Забраковано серий / писем",
            f"{int(mfg_row['explicit_defects_count'])} / {int(mfg_row['unique_letters'])}",
            f"Глубина: {mfg_row['series_per_incident']} сер./письмо",
        )
        c4.metric(
            "Главная технологическая проблема",
            f"{int(mfg_row['primary_defect_batches'])} сер.",
            mfg_row["primary_defect_category"],
        )

        r_left, r_right = st.columns([0.9, 1.1])
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
                title=f"Профиль дефектов: {selected_mfg}",
                height=380,
                margin=dict(l=40, r=40, t=50, b=30),
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
                height=350,
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

        st.markdown("#### 🔁 Витрина хронических серийных рецидивов по препаратам (`v_chronic_drug_recidivism`)")
        st.caption(
            "Рассчитано в SQL через оконные функции `LAG(letter_date) OVER (PARTITION BY manufacturer, drug, defect)` "
            "и `DENSE_RANK()`. Позволяет отличить разовый отзыв партии от многолетней неотвалидированной технологии."
        )
        st.dataframe(
            df_chronic.head(25)[
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
            height=360,
        )

        st.markdown("#### 🔬 Экспертный разбор реальных R&D-кейсов из реестра Росздравнадзора (комментарий технолога)")
        e1, e2 = st.columns(2)
        with e1:
            with st.expander("1️⃣ Кейс «Растворение» ТЛФ: Спиронолактон (32 серии), Ибупрофен Велфарм, Танацехол (14 серий)", expanded=True):
                st.markdown(
                    """
                    - **Наблюдение в данных:** 547 забракованных серий ТЛФ по тесту «Растворение» и «Распадаемость» (`20.0%` всех дефектов таблеток и капсул).
                    - **Технологическая причина (R&D):**
                      1. **Смена поставщика или гранулометрии гидрофобной АФС:** для плохо растворимых субстанций (BCS Class II — спиронолактон, ибупрофен) укрупнение размера частиц $d_{90}$ или изменение полиморфной модификации без пересмотра количества супердезинтегранта (кроскармеллозы натрия / натрия крахмала гликолята) и режима влажной грануляции обрушивает профиль высвобождения за 45 минут.
                      2. **Перепрессовка и старение оболочки:** избыточное усилие прессования на роторном таблетпрессе или сшивка (cross-linking) желатина капсул в присутствии следов альдегидов во вспомогательных веществах.
                    """
                )
            with st.expander("2️⃣ Кейс «Родственные примеси»: Эналаприл (108 серий в 47 письмах) и Капотен (каптоприла дисульфид)", expanded=True):
                st.markdown(
                    """
                    - **Наблюдение в данных:** у ПАО «Биосинтез» по препарату «Эналаприл» вышло **47 отдельных писем РЗН** (108 серий) по показателю «Родственные примеси».
                    - **Химико-технологическая причина:**
                      - Молекула эналаприла малеата крайне чувствительна к влаге и кислотности микроокружения таблетки: идут две конкурирующие реакции деградации — гидролиз до **эналаприлата** и внутримолекулярная циклизация до **дикетопиперазинового производного**.
                      - В «Капотене» (каптоприл) свободная тиольная группа $-\\text{SH}$ легко окисляется кислородом воздуха в присутствии следов ионов тяжелых металлов до **дисульфида каптоприла**. Требуется жесткий контроль остаточной влаги гранулята (LOD) и барьерных свойств блистера (ПВХ/ПВДХ или Ал/Ал).
                    """
                )
        with e2:
            with st.expander("3️⃣ Кейс гидролиза АСК: Цитрамон П (27 серий в 25 письмах — свободная салициловая кислота)", expanded=True):
                st.markdown(
                    """
                    - **Наблюдение в данных:** хронический отзыв серий «Цитрамона П» по комбинации показателей `Описание; Родственные примеси (салициловая кислота)` и `Количественное определение`.
                    - **Химико-технологическая причина:**
                      - В присутствии свободной влаги (из крахмала или при негерметичной контурной ячейковой упаковке) ацетилсалициловая кислота подвергается автокаталитическому эфирному гидролизу с отщеплением уксусной кислоты и кристаллизацией **салициловой кислоты** (игольчатые кристаллы на поверхности таблетки — отсюда одновременный брак по «Описанию» и «Родственным примесям»).
                    """
                )
            with st.expander("4️⃣ Кейс парентеральных форм: Механические включения (153 серии инъекционных растворов)", expanded=True):
                st.markdown(
                    """
                    - **Наблюдение в данных:** у производителей ампульных и инфузионных препаратов (`ФКП «Армавирская биофабрика»`, `ОАО «Самарамедпром»` и др.) `Механические включения` и `Прозрачность/Цветность` формируют ядро технологического брака.
                    - **Технологическая причина:**
                      - Нарушение целостности мембранных стерилизующих фильтров ($0.22\\ \\mu\\text{m}$), образование сколов стекла при запайке ампул газовыми горелками или выщелачивание/деламинация боросиликатного стекла при автоклавировании растворов со щелочным pH.
                    """
                )

    # =========================================================================
    # ВКЛАДКА 4: ПРОВЕРКА СТАТ. ГИПОТЕЗ И МЕТОДОЛОГИЯ РАСЧЕТА PRRI
    # =========================================================================
    with tab4:
        st.markdown("### 📐 Математическая модель индекса PRRI (Pharma Recall Risk Index)")
        st.latex(
            r"\text{PRRI} = \sum_{i=1}^{N} \left( W_{\text{defect}, i} \times W_{\text{action}, i} \times W_{\text{chronic}, i} \times e^{-\lambda \cdot \Delta t_i} \right)"
        )
        m_col1, m_col2 = st.columns([1.15, 0.85])
        with m_col1:
            st.markdown(
                """
                1. **$W_{\\text{defect}, i}$ — Доменный вес критичности дефекта (классификация химика-аналитика):**
                   - **Level 1 (`1.0`) — Упаковка и логистика:** *Маркировка, Упаковка, Комплектность*. Не несет фармакологического риска для пациента.
                   - **Level 2 (`2.5`) — Технологический брак формы:** *Растворение, Распадаемость, Однородность дозирования, Средняя масса, Описание, Механические включения, pH, Осмоляльность*. Сбой техпроцесса производства формы.
                   - **Level 3 (`5.0`) — Критический брак АФС и микробиологии:** *Родственные и посторонние примеси, Количественное определение, Подлинность, Стерильность, Микробиологическая чистота, Бактериальные эндотоксины, Пирогенность*. Прямая угроза токсичности или потери эффективности.
                2. **$W_{\\text{action}, i}$ — Строгость регуляторной меры РЗН:**
                   - `Отзыв` / `Изъятие` / `Прекращение обращения` = **$1.5$**; `Приостановление` / `Информационное` = **$1.0$**; `Разрешение реализации` = **$0.0$** (исключено из брака).
                3. **$W_{\\text{chronic}, i} = 1 + 0.25 \\cdot \\ln(N_{\\text{mfg, cat}})$ — Фактор системности / серийности:**
                   - Повышающий множитель для дефектов Level 2 и Level 3 при повторении однотипного брака у завода на $N > 1$ сериях.
                4. **$e^{-\\lambda \\cdot \\Delta t_i}$ — Временное затухание (Recency Decay):**
                   - Свежие забраковки за последние 12 месяцев имеют максимальный вес **$1.0$**, более старые экспоненциально затухают с $\\lambda = 0.50\\text{ год}^{-1}$.
                """
            )
        with m_col2:
            st.markdown("#### 🧮 Интерактивный калькулятор-симулятор PRRI")
            sim_level = st.selectbox(
                "Уровень критичности дефекта:",
                options=[
                    ("Level 3: Примеси / Колич. определение / Микробиология (5.0)", 5.0),
                    ("Level 2: Растворение / Однородность / Описание (2.5)", 2.5),
                    ("Level 1: Маркировка и упаковка (1.0)", 1.0),
                ],
                format_func=lambda x: x[0],
            )
            sim_action = st.radio(
                "Регуляторная мера РЗН:",
                options=[("Отзыв / Изъятие (1.5)", 1.5), ("Приостановление (1.0)", 1.0)],
                format_func=lambda x: x[0],
                horizontal=True,
            )
            sim_batches = st.slider("Число серий с этим дефектом у завода (N):", 1, 50, 5)
            sim_age = st.slider("Возраст письма РЗН (лет):", 0.0, 6.0, 0.5, 0.5)

            w_def_sim = sim_level[1]
            w_act_sim = sim_action[1]
            w_chr_sim = 1.0 + 0.25 * np.log(sim_batches) if (w_def_sim > 1.0 and sim_batches > 1) else 1.0
            decay_sim = float(np.exp(-0.50 * max(0.0, sim_age - 1.0)))
            single_score = w_def_sim * w_act_sim * w_chr_sim * decay_sim
            batch_total_score = single_score * sim_batches

            st.success(
                f"**Вес 1 серии:** `{single_score:.2f}` "
                f"($W_{{\\text{{def}}}}={w_def_sim} \\times W_{{\\text{{act}}}}={w_act_sim} "
                f"\\times W_{{\\text{{chr}}}}={w_chr_sim:.2f} \\times \\text{{Decay}}={decay_sim:.2f}$)  \n"
                f"**Суммарный вклад {sim_batches} серий в PRRI завода:** **`{batch_total_score:.2f}`**"
            )

        st.divider()
        st.markdown("### 🧪 Проверка статистических гипотез (`scipy.stats`, уровень значимости $\\alpha = 0.05$)")
        hyp = load_hypothesis_results()

        for key in (
            "h1_origin_vs_defect_level",
            "h2_dosage_form_vs_tdr",
            "h3_prri_mann_whitney",
            "h4_seasonality_kruskal",
        ):
            item = hyp[key]
            badge = "✅ H0 ОТВЕРГНУТА (различия статистически значимы)" if item["reject_h0"] else "⚪ H0 НЕ ОТВЕРГНУТА (значимых различий нет)"
            with st.container(border=True):
                st.markdown(f"#### {item['title']}")
                st.caption(f"**Метод:** {item['test_name']} | **Статус:** {badge}")
                hc1, hc2, hc3 = st.columns(3)
                hc1.metric(f"Статистика ({item['stat_name']})", f"{item['stat_value']}")
                hc2.metric("p-value", f"{item['p_value']:.4e}" if item["p_value"] < 0.001 else f"{item['p_value']:.4f}")
                hc3.metric(f"Размер эффекта ({item['effect_size_name']})", f"{item['effect_size_value']}")

                st.markdown(f"- **$H_0$:** {item['h0']}  \n- **$H_1$:** {item['h1']}")
                st.markdown(f"**Аналитический и доменный вывод:** {item['conclusion']}")

                if "contingency_pct" in item:
                    st.dataframe(item["contingency_pct"], width="stretch")
                if "summary_table" in item:
                    st.dataframe(item["summary_table"], width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
