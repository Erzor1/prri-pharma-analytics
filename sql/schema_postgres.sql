-- ============================================================================
-- PostgreSQL 16 Schema DDL — Мониторинг и скоринг брака ЛС (PRRI)
-- Используется при развертывании в контейнере PostgreSQL (docker-compose)
-- ============================================================================

CREATE TABLE IF NOT EXISTS dim_manufacturers (
    manufacturer_clean        TEXT PRIMARY KEY,
    origin_type               TEXT NOT NULL,
    country                   TEXT,
    total_recalls             INTEGER NOT NULL DEFAULT 0,
    unique_letters            INTEGER NOT NULL DEFAULT 0,
    unique_drugs              INTEGER NOT NULL DEFAULT 0,
    explicit_defects_count    INTEGER NOT NULL DEFAULT 0,
    level1_packaging_count    INTEGER NOT NULL DEFAULT 0,
    level2_tech_count         INTEGER NOT NULL DEFAULT 0,
    level3_critical_count     INTEGER NOT NULL DEFAULT 0,
    level0_unspecified_count  INTEGER NOT NULL DEFAULT 0,
    recent_12m_recalls        INTEGER NOT NULL DEFAULT 0,
    prri_score                NUMERIC(12, 2) NOT NULL DEFAULT 0.00,
    prri_total_score          NUMERIC(12, 2) NOT NULL DEFAULT 0.00,
    avg_incident_severity     NUMERIC(6, 2) NOT NULL DEFAULT 0.00,
    max_chronic_factor        NUMERIC(6, 2) NOT NULL DEFAULT 1.00,
    last_recall_date          DATE,
    series_per_incident       NUMERIC(8, 2) NOT NULL DEFAULT 1.00,
    tdr_pct                   NUMERIC(5, 1) NOT NULL DEFAULT 0.0,
    risk_tier                 TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS fact_recalls (
    letter_id                   TEXT,
    trade_name                  TEXT NOT NULL,
    dosage_form_raw             TEXT,
    batch_number                TEXT,
    manufacturer_raw            TEXT,
    country                     TEXT,
    status                      TEXT,
    regulatory_action           TEXT,
    scope                       TEXT,
    letter_raw                  TEXT,
    letter_link                 TEXT,
    defect_parameter            TEXT,
    reg_number                  TEXT,
    letter_number               TEXT,
    letter_date                 DATE,
    year                        INTEGER,
    month                       INTEGER,
    quarter                     INTEGER,
    year_month                  TEXT,
    age_days                    INTEGER,
    age_years                   NUMERIC(8, 4),
    origin_type                 TEXT,
    manufacturer_clean          TEXT REFERENCES dim_manufacturers(manufacturer_clean),
    dosage_form_group           TEXT,
    dosage_form_short           TEXT,
    defect_level_num            INTEGER,
    defect_level                TEXT,
    w_defect                    NUMERIC(6, 2),
    defect_category             TEXT,
    defect_params_count         INTEGER,
    has_explicit_defect         INTEGER,
    is_tech_or_critical_defect  INTEGER,
    w_action                    NUMERIC(6, 2),
    is_recall_event             INTEGER,
    mfg_same_defect_batches     INTEGER,
    w_chronic                   NUMERIC(8, 4),
    recency_decay               NUMERIC(8, 4),
    prri_total_incident_score   NUMERIC(12, 4),
    prri_incident_score         NUMERIC(12, 4)
);

CREATE INDEX IF NOT EXISTS idx_fact_recalls_date ON fact_recalls (letter_date);
CREATE INDEX IF NOT EXISTS idx_fact_recalls_year_month ON fact_recalls (year_month);
CREATE INDEX IF NOT EXISTS idx_fact_recalls_mfg ON fact_recalls (manufacturer_clean);
CREATE INDEX IF NOT EXISTS idx_fact_recalls_form_group ON fact_recalls (dosage_form_group);
CREATE INDEX IF NOT EXISTS idx_fact_recalls_defect_cat ON fact_recalls (defect_category);
