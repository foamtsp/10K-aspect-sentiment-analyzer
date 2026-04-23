"""Unified scorer Lambda behind API Gateway (HTTP API).

Routes:
    GET  /tickers/{ticker}/filings                        → list recent 10-Ks
    POST /tickers/{ticker}/analysis?accession=...         → start async scoring
                                                            returns 200 (cached)
                                                            or 202 (started)
    GET  /tickers/{ticker}/analysis?accession=...         → poll cache
                                                            returns 200 (ready)
                                                            or 202 (pending)

Async flow:
    POST handler checks the MySQL cache. On a miss it invokes this same Lambda
    with InvocationType=Event (fire-and-forget) and returns 202 immediately.
    The worker invocation runs the full pipeline and writes to the cache.
    The UI polls GET every 5 s until the cache row appears.
"""
from __future__ import annotations

import json
import os
import pickle
from typing import Any

import boto3
import numpy as np
import xgboost as xgb

from common.aggregation import aggregate
from common.aspects import ASPECTS, tag_batch
from common.db import (
    connect,
    get_cached,
    previous_aspect_scores,
    upsert_scored_filing,
)
from common.edgar import cik_for, fetch_filing_html, list_10ks
from common.sections import parse_filing

FINBERT_MODEL_ID = os.environ.get("FINBERT_MODEL_ID", "JanhaviS14/finance-sentiment-mini-finbert")
FINBERT_BATCH = int(os.environ.get("FINBERT_BATCH_SIZE", "64"))
FINBERT_MAX_LEN = 128  # sentences rarely exceed 128 tokens; halves inference time
MODEL_DIR = os.environ.get("MODEL_DIR", "/opt/model")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "v1")
HORIZON_DAYS = int(os.environ.get("HORIZON_DAYS", "30"))
DEFAULT_SIC = int(os.environ.get("DEFAULT_SIC", "7370"))
FUNCTION_NAME = os.environ.get("AWS_LAMBDA_FUNCTION_NAME", "")

_tokenizer: Any = None
_finbert: Any = None
_torch: Any = None
_id2label: dict[int, str] = {}  # populated after model load; keys normalised to lowercase
_booster: xgb.Booster | None = None
_calibrator = None
_feature_cols: list[str] | None = None
_lambda_client = None


def _get_lambda_client():
    global _lambda_client
    if _lambda_client is None:
        _lambda_client = boto3.client("lambda")
    return _lambda_client


def _load_finbert():
    global _tokenizer, _finbert, _torch, _id2label
    if _finbert is None:
        # Lazy-import torch/transformers so cold starts on the lightweight POST
        # handler don't pay the ~15-20s model-load penalty.
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        _torch = torch
        _tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL_ID)
        _finbert = AutoModelForSequenceClassification.from_pretrained(
            FINBERT_MODEL_ID
        ).eval()
        # Normalise labels to lowercase so any model (ProsusAI, nickmuchi, etc.)
        # produces keys our aggregation code expects: positive/negative/neutral.
        _id2label = {i: lbl.lower() for i, lbl in _finbert.config.id2label.items()}
        _torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "4")))
        print(f"[finbert] loaded {FINBERT_MODEL_ID}, labels={_id2label}")
    return _tokenizer, _finbert


def _invoke_finbert(sentences: list[str]) -> list[dict[str, float]]:
    if not sentences:
        return []
    tokenizer, model = _load_finbert()
    results: list[dict[str, float]] = []
    with _torch.no_grad():
        for i in range(0, len(sentences), FINBERT_BATCH):
            chunk = sentences[i : i + FINBERT_BATCH]
            enc = tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=FINBERT_MAX_LEN,
                return_tensors="pt",
            )
            probs = _torch.softmax(model(**enc).logits, dim=-1).numpy()
            for p in probs:
                results.append({_id2label[j]: float(p[j]) for j in range(len(p))})
    return results


def _load_model() -> tuple[Any, Any, list[str]]:
    global _booster, _calibrator, _feature_cols
    if _booster is None:
        _booster = xgb.Booster()
        _booster.load_model(os.path.join(MODEL_DIR, "booster.json"))
        with open(os.path.join(MODEL_DIR, "calibrator.pkl"), "rb") as f:
            _calibrator = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "feature_cols.json")) as f:
            _feature_cols = json.load(f)
    return _booster, _calibrator, _feature_cols


def _json(status: int, body: dict) -> dict:
    return {
        "statusCode": status,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(body, default=str),
    }


def _row_to_analysis(row: dict) -> dict:
    return {
        "accession": row["accession"],
        "ticker": row["ticker"],
        "cik": row["cik"],
        "filing_date": str(row["filing_date"]),
        "form": row["form"],
        "prediction": row["prediction"],
        "probability_up": float(row["probability_up"]),
        "horizon_days": int(row["horizon_days"]),
        "model_version": row["model_version"],
        "n_sentences": int(row["n_sentences"]),
        "aspects": {a: float(row[a]) for a in ASPECTS},
        "deltas": {
            a: (float(row[f"{a}_delta"]) if row.get(f"{a}_delta") is not None else None)
            for a in ASPECTS
        },
    }


def _score_filing(conn, ticker: str, cik: int, filing) -> dict:
    html = fetch_filing_html(cik, filing.accession, filing.primary_doc)
    parsed = parse_filing(
        html,
        cik=str(cik),
        ticker=ticker,
        filing_date=filing.filing_date,
        accession=filing.accession,
    )

    flat_sentences: list[str] = []
    flat_sections: list[str] = []
    for section, sents in parsed.sections.items():
        for s in sents:
            flat_sentences.append(s)
            flat_sections.append(section)

    tags = tag_batch(flat_sentences, flat_sections)
    keep_idx = [i for i, t in enumerate(tags) if t]
    kept_sentences = [flat_sentences[i] for i in keep_idx]
    kept_sections = [flat_sections[i] for i in keep_idx]

    sentiments = _invoke_finbert(kept_sentences)
    vector = aggregate(kept_sentences, sentiments, kept_sections)

    prev = previous_aspect_scores(conn, ticker, filing.filing_date) or {}
    deltas: dict[str, float | None] = {}
    feature_row: dict[str, float] = {}
    for a in ASPECTS:
        feature_row[f"{a}_score"] = vector.scores[a]
        feature_row[f"{a}_count"] = float(vector.counts[a])
        if prev:
            d = vector.scores[a] - float(prev.get(a, 0.0))
            feature_row[f"{a}_delta"] = d
            deltas[a] = d
        else:
            feature_row[f"{a}_delta"] = 0.0
            deltas[a] = None
    feature_row["sic"] = float(DEFAULT_SIC)
    feature_row["n_sentences"] = float(len(flat_sentences))

    _, calibrator, cols = _load_model()
    x = np.array([[feature_row[c] for c in cols]], dtype=np.float32)
    # calibrator is a CalibratedClassifierCV wrapping XGBoost — pass the full
    # feature matrix; predict_proba returns [p_down, p_up].
    prob_up = float(calibrator.predict_proba(x)[0, 1])
    prediction = "up" if prob_up >= 0.5 else "down"

    row = {
        "accession": filing.accession,
        "ticker": ticker,
        "cik": str(cik),
        "filing_date": filing.filing_date,
        "form": filing.form,
        **{a: vector.scores[a] for a in ASPECTS},
        **{f"{a}_delta": deltas[a] for a in ASPECTS},
        "probability_up": prob_up,
        "prediction": prediction,
        "horizon_days": HORIZON_DAYS,
        "model_version": MODEL_VERSION,
        "n_sentences": len(flat_sentences),
    }
    upsert_scored_filing(conn, row)
    return _row_to_analysis(get_cached(conn, filing.accession) or row)


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------

def _handle_filings(ticker: str) -> dict:
    cik = cik_for(ticker)
    if cik is None:
        return _json(404, {"error": f"unknown ticker {ticker}"})
    filings = list_10ks(cik, limit=10)
    return _json(
        200,
        {
            "ticker": ticker,
            "cik": cik,
            "filings": [
                {
                    "accession": f.accession,
                    "filing_date": f.filing_date,
                    "form": f.form,
                    "primary_doc": f.primary_doc,
                }
                for f in filings
            ],
        },
    )


def _handle_analysis_trigger(ticker: str, accession: str | None, force: bool) -> dict:
    """POST: return 200 if cached, otherwise fire async worker and return 202."""
    print(f"[trigger] ticker={ticker} accession={accession} force={force}")
    if not accession:
        return _json(400, {"error": "accession required"})

    print(f"[trigger] resolving CIK for {ticker}")
    cik = cik_for(ticker)
    if cik is None:
        print(f"[trigger] unknown ticker {ticker}")
        return _json(404, {"error": f"unknown ticker {ticker}"})
    print(f"[trigger] CIK={cik}")

    with connect() as conn:
        if not force:
            print("[trigger] checking cache")
            cached = get_cached(conn, accession)
            if cached:
                print("[trigger] cache hit — returning 200")
                return _json(200, {"cached": True, "analysis": _row_to_analysis(cached)})
            print("[trigger] cache miss")

    print(f"[trigger] invoking async worker function={FUNCTION_NAME}")
    _get_lambda_client().invoke(
        FunctionName=FUNCTION_NAME,
        InvocationType="Event",
        Payload=json.dumps({
            "_worker": True,
            "ticker": ticker,
            "accession": accession,
            "force": force,
        }).encode(),
    )
    print("[trigger] worker fired — returning 202")
    return _json(202, {"status": "pending", "accession": accession})


def _handle_analysis_poll(ticker: str, accession: str | None) -> dict:
    """GET: return 200 with result if cached, otherwise 202 pending."""
    if not accession:
        return _json(400, {"error": "accession required"})
    with connect() as conn:
        cached = get_cached(conn, accession)
        if cached:
            return _json(200, {"cached": True, "analysis": _row_to_analysis(cached)})
    return _json(202, {"status": "pending", "accession": accession})


def _run_worker(ticker: str, accession: str, force: bool) -> None:
    """Async worker: full scoring pipeline, writes result to MySQL cache."""
    print(f"[worker] start ticker={ticker} accession={accession} force={force}")
    cik = cik_for(ticker)
    if cik is None:
        print(f"[worker] unknown ticker {ticker} — aborting")
        return
    print(f"[worker] CIK={cik}, fetching filings")
    filings = list_10ks(cik, limit=10)
    filing = next((f for f in filings if f.accession == accession), None)
    if filing is None:
        print(f"[worker] accession {accession} not found in recent 10-Ks — aborting")
        return
    print(f"[worker] filing found: {filing.filing_date}")
    with connect() as conn:
        if not force:
            if get_cached(conn, accession):
                print("[worker] already cached by another invocation — skipping")
                return
        print("[worker] starting _score_filing (EDGAR fetch + FinBERT + XGBoost)")
        _score_filing(conn, ticker, cik, filing)
        print("[worker] done — result written to cache")


def handler(event: dict, _context) -> dict:
    # Worker mode: invoked async by _handle_analysis_trigger.
    if event.get("_worker"):
        try:
            _run_worker(
                ticker=event["ticker"].upper(),
                accession=event["accession"],
                force=bool(event.get("force", False)),
            )
        except Exception as exc:  # noqa: BLE001
            import traceback
            print(f"[worker] UNHANDLED ERROR: {exc}")
            print(traceback.format_exc())
        return {}

    path = event.get("rawPath") or event.get("path") or ""
    method = (
        event.get("requestContext", {}).get("http", {}).get("method")
        or event.get("httpMethod")
        or "GET"
    ).upper()
    params = event.get("pathParameters") or {}
    qs = event.get("queryStringParameters") or {}
    ticker = (params.get("ticker") or "").upper()

    if not ticker:
        return _json(400, {"error": "ticker required"})

    try:
        if method == "GET" and path.endswith("/filings"):
            return _handle_filings(ticker)
        if method == "POST" and path.endswith("/analysis"):
            return _handle_analysis_trigger(
                ticker,
                accession=qs.get("accession"),
                force=str(qs.get("refresh", "")).lower() in ("1", "true", "yes"),
            )
        if method == "GET" and path.endswith("/analysis"):
            return _handle_analysis_poll(ticker, accession=qs.get("accession"))
    except Exception as exc:  # noqa: BLE001
        return _json(500, {"error": str(exc)})

    return _json(404, {"error": f"no route for {method} {path}"})
