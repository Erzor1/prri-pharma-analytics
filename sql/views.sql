-- ============================================================================
-- Аналитические витрины данных проекта PRRI (Pharma Recall Risk Index)
-- Совместимость: PostgreSQL 15/16+ и SQLite 3.35+ (ANSI SQL, CTE, Window Functions)
-- ============================================================================

-- ----------------------------------------------------------------------------
-- ВИТРИНА 1: v_monthly_defect_dynamics
-- Помесячная динамика забраковок, MoM-прирост (LAG), 3-месячное скользящее
-- среднее (AVG OVER ROWS) и накопительный итог с начала года (YTD SUM OVER).
-- ----------------------------------------------------------------------------
DROP VIEW IF EXISTS v_monthly_defect_dynamics;

CREATE VIEW v_monthly_defect_dynamics AS
WITH monthly_agg AS (
    SELECT
        year_month,
        year,
        month,
        COUNT(*) AS total_recall_rows,
        COUNT(DISTINCT letter_number) AS unique_letters,
        SUM(CASE WHEN has_explicit_defect = 1 THEN 1 ELSE 0 END) AS verified_defects,
        SUM(CASE WHEN defect_level_num = 1 THEN 1 ELSE 0 END) AS level1_packaging,
        SUM(CASE WHEN defect_level_num = 2 THEN 1 ELSE 0 END) AS level2_technological,
        SUM(CASE WHEN defect_level_num = 3 THEN 1 ELSE 0 END) AS level3_critical,
        SUM(CASE WHEN defect_level_num = 0 THEN 1 ELSE 0 END) AS level0_precautionary,
        ROUND(SUM(prri_incident_score), 2) AS monthly_prri_score,
        ROUND(
            SUM(CASE WHEN defect_level_num IN (2, 3) THEN 1.0 ELSE 0.0 END) * 100.0
            / NULLIF(SUM(CASE WHEN has_explicit_defect = 1 THEN 1 ELSE 0 END), 0),
            1
        ) AS monthly_tdr_pct
    FROM fact_recalls
    WHERE is_recall_event = 1
    GROUP BY year_month, year, month
),
window_metrics AS (
    SELECT
        m.*,
        LAG(verified_defects, 1) OVER (ORDER BY year_month) AS prev_month_verified,
        verified_defects - LAG(verified_defects, 1) OVER (ORDER BY year_month) AS mom_verified_diff,
        ROUND(
            (verified_defects - LAG(verified_defects, 1) OVER (ORDER BY year_month)) * 100.0
            / NULLIF(LAG(verified_defects, 1) OVER (ORDER BY year_month), 0),
            1
        ) AS mom_verified_growth_pct,
        ROUND(
            AVG(verified_defects * 1.0) OVER (
                ORDER BY year_month
                ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
            ),
            1
        ) AS ma3_verified_defects,
        ROUND(
            AVG(monthly_prri_score * 1.0) OVER (
                ORDER BY year_month
                ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
            ),
            2
        ) AS ma3_prri_score,
        SUM(verified_defects) OVER (
            PARTITION BY year
            ORDER BY year_month
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS ytd_verified_defects
    FROM monthly_agg m
)
SELECT *
FROM window_metrics
ORDER BY year_month;


-- ----------------------------------------------------------------------------
-- ВИТРИНА 2: v_manufacturer_risk_ranking
-- Рейтинг фармпроизводителей по PRRI с оконным ранжированием (DENSE_RANK),
-- расчетом кривой Парето (кумулятивная доля риска рынка SUM OVER) и
-- определением главной («узкой») проблемы завода через ROW_NUMBER().
-- ----------------------------------------------------------------------------
DROP VIEW IF EXISTS v_manufacturer_risk_ranking;

CREATE VIEW v_manufacturer_risk_ranking AS
WITH defect_mode_counts AS (
    SELECT
        manufacturer_clean,
        defect_category,
        COUNT(*) AS category_batches,
        ROUND(SUM(prri_incident_score), 2) AS category_prri
    FROM fact_recalls
    WHERE is_recall_event = 1
      AND has_explicit_defect = 1
    GROUP BY manufacturer_clean, defect_category
),
defect_mode_ranked AS (
    SELECT
        manufacturer_clean,
        defect_category AS primary_defect_category,
        category_batches AS primary_defect_batches,
        ROW_NUMBER() OVER (
            PARTITION BY manufacturer_clean
            ORDER BY category_batches DESC, category_prri DESC
        ) AS rn
    FROM defect_mode_counts
),
top_defect_per_mfg AS (
    SELECT
        manufacturer_clean,
        primary_defect_category,
        primary_defect_batches
    FROM defect_mode_ranked
    WHERE rn = 1
)
SELECT
    d.manufacturer_clean,
    d.origin_type,
    d.country,
    d.risk_tier,
    d.prri_score,
    d.prri_total_score,
    d.tdr_pct,
    d.series_per_incident,
    d.total_recalls,
    d.explicit_defects_count,
    d.unique_letters,
    d.unique_drugs,
    d.level1_packaging_count,
    d.level2_tech_count,
    d.level3_critical_count,
    d.level0_unspecified_count,
    d.recent_12m_recalls,
    d.max_chronic_factor,
    d.last_recall_date,
    COALESCE(t.primary_defect_category, 'Превентивный отзыв (без показателя)') AS primary_defect_category,
    COALESCE(t.primary_defect_batches, 0) AS primary_defect_batches,
    DENSE_RANK() OVER (ORDER BY d.prri_score DESC) AS overall_risk_rank,
    DENSE_RANK() OVER (
        PARTITION BY d.origin_type
        ORDER BY d.prri_score DESC
    ) AS origin_risk_rank,
    ROUND(
        d.prri_score * 100.0 / NULLIF(SUM(d.prri_score) OVER (), 0),
        2
    ) AS market_risk_share_pct,
    ROUND(
        SUM(d.prri_score) OVER (
            ORDER BY d.prri_score DESC, d.manufacturer_clean
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) * 100.0 / NULLIF(SUM(d.prri_score) OVER (), 0),
        2
    ) AS cumulative_pareto_risk_pct
FROM dim_manufacturers d
LEFT JOIN top_defect_per_mfg t
    ON d.manufacturer_clean = t.manufacturer_clean
ORDER BY d.prri_score DESC, d.prri_total_score DESC;


-- ----------------------------------------------------------------------------
-- ВИТРИНА 3: v_dosage_form_defect_matrix
-- Технологическая R&D-матрица «Лекарственная форма × Категория брака»
-- с расчетом долей внутри каждой технологической группы (SUM OVER PARTITION)
-- и ранжированием дефектов внутри лекарственной формы (ROW_NUMBER).
-- ----------------------------------------------------------------------------
DROP VIEW IF EXISTS v_dosage_form_defect_matrix;

CREATE VIEW v_dosage_form_defect_matrix AS
WITH form_defect_agg AS (
    SELECT
        dosage_form_group,
        defect_level_num,
        defect_level,
        defect_category,
        COUNT(*) AS batches_count,
        COUNT(DISTINCT trade_name) AS affected_drugs,
        COUNT(DISTINCT manufacturer_clean) AS affected_manufacturers,
        ROUND(SUM(prri_incident_score), 2) AS total_prri_score
    FROM fact_recalls
    WHERE is_recall_event = 1
      AND has_explicit_defect = 1
    GROUP BY
        dosage_form_group,
        defect_level_num,
        defect_level,
        defect_category
)
SELECT
    f.dosage_form_group,
    f.defect_level_num,
    f.defect_level,
    f.defect_category,
    f.batches_count,
    f.affected_drugs,
    f.affected_manufacturers,
    f.total_prri_score,
    SUM(f.batches_count) OVER (PARTITION BY f.dosage_form_group) AS form_group_total_batches,
    ROUND(
        f.batches_count * 100.0
        / NULLIF(SUM(f.batches_count) OVER (PARTITION BY f.dosage_form_group), 0),
        2
    ) AS share_within_form_pct,
    ROW_NUMBER() OVER (
        PARTITION BY f.dosage_form_group
        ORDER BY f.batches_count DESC, f.total_prri_score DESC
    ) AS defect_rank_in_form
FROM form_defect_agg f
ORDER BY form_group_total_batches DESC, defect_rank_in_form ASC;


-- ----------------------------------------------------------------------------
-- ВИТРИНА 4: v_chronic_drug_recidivism
-- R&D-витрина хронических технологических проблем и рецидивов по препаратам:
-- отслеживает повторные письма РЗН по одной и той же паре (Завод + Препарат + Дефект)
-- с использованием LAG(letter_date) и считает глубину поражения (Series-per-Incident).
-- ----------------------------------------------------------------------------
DROP VIEW IF EXISTS v_chronic_drug_recidivism;

CREATE VIEW v_chronic_drug_recidivism AS
WITH letter_level_events AS (
    SELECT
        manufacturer_clean,
        origin_type,
        trade_name,
        dosage_form_group,
        dosage_form_short,
        defect_level,
        defect_category,
        letter_number,
        MIN(letter_date) AS letter_date,
        COUNT(*) AS batches_in_letter,
        ROUND(SUM(prri_incident_score), 2) AS letter_prri_score
    FROM fact_recalls
    WHERE is_recall_event = 1
      AND is_tech_or_critical_defect = 1
    GROUP BY
        manufacturer_clean,
        origin_type,
        trade_name,
        dosage_form_group,
        dosage_form_short,
        defect_level,
        defect_category,
        letter_number
),
sequenced_events AS (
    SELECT
        e.*,
        ROW_NUMBER() OVER (
            PARTITION BY manufacturer_clean, trade_name, defect_category
            ORDER BY letter_date, letter_number
        ) AS letter_seq_num,
        LAG(letter_date, 1) OVER (
            PARTITION BY manufacturer_clean, trade_name, defect_category
            ORDER BY letter_date, letter_number
        ) AS prev_letter_date
    FROM letter_level_events e
),
chronic_summary AS (
    SELECT
        manufacturer_clean,
        origin_type,
        trade_name,
        dosage_form_group,
        dosage_form_short,
        defect_level,
        defect_category,
        COUNT(DISTINCT letter_number) AS unique_letters_count,
        SUM(batches_in_letter) AS total_recalled_batches,
        MAX(batches_in_letter) AS max_batches_in_single_letter,
        ROUND(
            SUM(batches_in_letter) * 1.0 / NULLIF(COUNT(DISTINCT letter_number), 0),
            2
        ) AS series_per_letter_ratio,
        MIN(letter_date) AS first_incident_date,
        MAX(letter_date) AS last_incident_date,
        ROUND(SUM(letter_prri_score), 2) AS drug_defect_prri_score
    FROM sequenced_events
    GROUP BY
        manufacturer_clean,
        origin_type,
        trade_name,
        dosage_form_group,
        dosage_form_short,
        defect_level,
        defect_category
    HAVING SUM(batches_in_letter) >= 2
)
SELECT
    c.*,
    DENSE_RANK() OVER (ORDER BY c.drug_defect_prri_score DESC) AS chronic_case_rank
FROM chronic_summary c
ORDER BY c.drug_defect_prri_score DESC, c.total_recalled_batches DESC;
