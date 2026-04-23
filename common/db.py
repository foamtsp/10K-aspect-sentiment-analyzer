"""MySQL connection + cache helpers for the scorer.

Single table `scored_filings` serves as both the prediction cache and the
YoY-delta source: when scoring a new filing we look up the same ticker's
most recent prior row and diff aspect scores against it.
"""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from typing import Any, Iterator

import boto3
import pymysql
from pymysql.cursors import DictCursor

from .aspects import ASPECTS


def _load_secret() -> dict[str, str]:
    secret_arn = os.environ.get("DB_SECRET_ARN")
    if not secret_arn:
        return {
            "host": os.environ["DB_HOST"],
            "port": os.environ.get("DB_PORT", "3306"),
            "username": os.environ["DB_USER"],
            "password": os.environ["DB_PASSWORD"],
            "dbname": os.environ.get("DB_NAME", "tenk"),
        }
    sm = boto3.client("secretsmanager")
    payload = json.loads(sm.get_secret_value(SecretId=secret_arn)["SecretString"])
    return {
        "host": payload["host"],
        "port": str(payload.get("port", 3306)),
        "username": payload["username"],
        "password": payload["password"],
        "dbname": payload.get("dbname", "tenk"),
    }


@contextmanager
def connect() -> Iterator[pymysql.connections.Connection]:
    cfg = _load_secret()
    conn = pymysql.connect(
        host=cfg["host"],
        port=int(cfg["port"]),
        user=cfg["username"],
        password=cfg["password"],
        database=cfg["dbname"],
        cursorclass=DictCursor,
        autocommit=False,
        charset="utf8mb4",
    )
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_cached(conn, accession: str) -> dict | None:
    cur = conn.cursor()
    cur.execute("SELECT * FROM scored_filings WHERE accession = %s", (accession,))
    row = cur.fetchone()
    return dict(row) if row else None


def previous_aspect_scores(conn, ticker: str, before_date: str) -> dict[str, float] | None:
    """Most recent prior filing's aspect scores for this ticker, for YoY deltas."""
    cols = ", ".join(ASPECTS)
    cur = conn.cursor()
    cur.execute(
        f"""
        SELECT {cols}
        FROM scored_filings
        WHERE ticker = %s AND filing_date < %s
        ORDER BY filing_date DESC
        LIMIT 1
        """,
        (ticker, before_date),
    )
    row = cur.fetchone()
    return dict(row) if row else None


def upsert_scored_filing(conn, row: dict[str, Any]) -> None:
    aspect_cols = list(ASPECTS)
    delta_cols = [f"{a}_delta" for a in ASPECTS]
    base_cols = [
        "accession", "ticker", "cik", "filing_date", "form",
        *aspect_cols,
        *delta_cols,
        "probability_up", "prediction", "horizon_days", "model_version", "n_sentences",
    ]
    placeholders = ", ".join(f"%({c})s" for c in base_cols)
    col_list = ", ".join(base_cols)
    updates = ", ".join(
        f"{c} = new.{c}" for c in base_cols if c != "accession"
    )
    sql = (
        f"INSERT INTO scored_filings ({col_list}) VALUES ({placeholders}) AS new "
        f"ON DUPLICATE KEY UPDATE {updates}"
    )
    cur = conn.cursor()
    cur.execute(sql, row)
