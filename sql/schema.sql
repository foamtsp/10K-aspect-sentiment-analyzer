-- MySQL 8.x schema for the 10-K signal pipeline (MVP).
-- Single table: serves as prediction cache AND YoY-delta lookup source.
-- Apply once against the reused churn-db RDS instance:
--   CREATE DATABASE IF NOT EXISTS tenk;
--   USE tenk;
--   SOURCE schema.sql;

CREATE TABLE IF NOT EXISTS scored_filings (
    accession       VARCHAR(32) PRIMARY KEY,
    ticker          VARCHAR(16) NOT NULL,
    cik             VARCHAR(16) NOT NULL,
    filing_date     DATE        NOT NULL,
    form            VARCHAR(16) NOT NULL,

    -- Aspect sentiment scores in [-1, 1], one column per aspect.
    revenue         FLOAT NOT NULL DEFAULT 0,
    cash_flow       FLOAT NOT NULL DEFAULT 0,
    margins         FLOAT NOT NULL DEFAULT 0,
    ebitda          FLOAT NOT NULL DEFAULT 0,
    future_plans    FLOAT NOT NULL DEFAULT 0,
    risk_factors    FLOAT NOT NULL DEFAULT 0,
    guidance        FLOAT NOT NULL DEFAULT 0,

    -- YoY deltas (score - prior_filing_score). NULL when no prior filing exists.
    revenue_delta      FLOAT NULL DEFAULT NULL,
    cash_flow_delta    FLOAT NULL DEFAULT NULL,
    margins_delta      FLOAT NULL DEFAULT NULL,
    ebitda_delta       FLOAT NULL DEFAULT NULL,
    future_plans_delta FLOAT NULL DEFAULT NULL,
    risk_factors_delta FLOAT NULL DEFAULT NULL,
    guidance_delta     FLOAT NULL DEFAULT NULL,

    -- Calibrated XGBoost output.
    probability_up  FLOAT       NOT NULL,
    prediction      ENUM('up','down') NOT NULL,
    horizon_days    SMALLINT    NOT NULL DEFAULT 30,
    model_version   VARCHAR(32) NOT NULL,

    n_sentences     INT         NOT NULL DEFAULT 0,
    scored_at       DATETIME    NOT NULL DEFAULT CURRENT_TIMESTAMP
                                ON UPDATE CURRENT_TIMESTAMP
);

-- Lookup index for YoY-delta: "most recent prior filing for this ticker".
CREATE INDEX idx_scored_ticker_date ON scored_filings (ticker, filing_date DESC);

-- Migration: add delta columns if they don't exist yet (idempotent via stored proc).
DROP PROCEDURE IF EXISTS _add_delta_cols;
DELIMITER $$
CREATE PROCEDURE _add_delta_cols()
BEGIN
    DECLARE CONTINUE HANDLER FOR SQLSTATE '42S21' BEGIN END;
    ALTER TABLE scored_filings
        ADD COLUMN revenue_delta      FLOAT NULL DEFAULT NULL,
        ADD COLUMN cash_flow_delta    FLOAT NULL DEFAULT NULL,
        ADD COLUMN margins_delta      FLOAT NULL DEFAULT NULL,
        ADD COLUMN ebitda_delta       FLOAT NULL DEFAULT NULL,
        ADD COLUMN future_plans_delta FLOAT NULL DEFAULT NULL,
        ADD COLUMN risk_factors_delta FLOAT NULL DEFAULT NULL,
        ADD COLUMN guidance_delta     FLOAT NULL DEFAULT NULL;
END$$
DELIMITER ;
CALL _add_delta_cols();
DROP PROCEDURE IF EXISTS _add_delta_cols;
