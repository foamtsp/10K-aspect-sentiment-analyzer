#!/usr/bin/env python3
"""
End-to-end API test suite.

Usage:
    python scripts/test_api.py <api-base-url> [ticker]

Example:
    python scripts/test_api.py https://xxxxx.execute-api.us-east-1.amazonaws.com AAPL

Tests:
    1. GET  /tickers/{ticker}/filings          — list 10-Ks
    2. GET  /tickers/{ticker}/analysis         — missing accession → 400
    3. GET  /tickers/{ticker}/analysis?accession=<bad> — unknown accession → 400/404
    4. POST /tickers/{ticker}/analysis         — missing accession → 400
    5. POST /tickers/{ticker}/analysis?accession=<real>
            cache miss  → 202 pending
            cache hit   → 200 (second POST on same accession)
    6. Poll GET until 200 or timeout           — async flow end-to-end
    7. GET  /tickers/INVALID_ZZZ/filings       — unknown ticker → 404
"""
from __future__ import annotations

import sys
import time
import json
import urllib.request
import urllib.error

POLL_INTERVAL = 10
POLL_TIMEOUT  = 300
TICKER        = "NVDA"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def request(method: str, url: str, timeout: int = 30) -> tuple[int, dict]:
    req = urllib.request.Request(url, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def ok(label: str) -> None:
    print(f"  PASS  {label}")


def fail(label: str, detail: str = "") -> None:
    print(f"  FAIL  {label}" + (f": {detail}" if detail else ""))
    sys.exit(1)


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        ok(label)
    else:
        fail(label, detail)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_filings(base: str, ticker: str) -> str:
    """Returns the first accession number for use in later tests."""
    print("\n[1] GET /tickers/{ticker}/filings")
    status, body = request("GET", f"{base}/tickers/{ticker}/filings")
    check("status 200", status == 200, f"got {status}")
    check("body has 'filings' list", isinstance(body.get("filings"), list),
          json.dumps(body)[:200])
    filings = body["filings"]
    check("at least one filing returned", len(filings) > 0, "empty list")
    first = filings[0]
    check("filing has accession", "accession" in first)
    check("filing has filing_date", "filing_date" in first)
    print(f"        found {len(filings)} filings; using {first['accession']}")
    return first["accession"]


def test_filings_unknown_ticker(base: str) -> None:
    print("\n[2] GET /tickers/INVALID_ZZZ9999/filings  (expect 404)")
    status, body = request("GET", f"{base}/tickers/INVALID_ZZZ9999/filings")
    check("status 404", status == 404, f"got {status}: {body}")


def test_analysis_get_no_accession(base: str, ticker: str) -> None:
    print("\n[3] GET /tickers/{ticker}/analysis  (no accession → 400)")
    status, body = request("GET", f"{base}/tickers/{ticker}/analysis")
    check("status 400", status == 400, f"got {status}: {body}")


def test_analysis_get_bad_accession(base: str, ticker: str) -> None:
    print("\n[4] GET /tickers/{ticker}/analysis?accession=BAD  (unknown → 202 pending or 400)")
    status, body = request("GET", f"{base}/tickers/{ticker}/analysis?accession=0000000000-00-000000")
    check("status 202 or 400", status in (202, 400), f"got {status}: {body}")


def test_analysis_post_no_accession(base: str, ticker: str) -> None:
    print("\n[5] POST /tickers/{ticker}/analysis  (no accession → 400)")
    status, body = request("POST", f"{base}/tickers/{ticker}/analysis")
    check("status 400", status == 400, f"got {status}: {body}")


def test_async_flow(base: str, ticker: str, accession: str) -> None:
    print(f"\n[6] POST /tickers/{ticker}/analysis?accession={accession}")
    print("     (async flow: expect 202 on first call, poll until 200)")

    status, body = request("POST",
        f"{base}/tickers/{ticker}/analysis?accession={accession}",
        timeout=35)

    if status == 200:
        print("     cache hit on POST — result already in DB, skipping poll")
        _validate_analysis_body(body)
        return

    check("POST returns 202", status == 202, f"got {status}: {body}")
    check("body has status=pending", body.get("status") == "pending",
          json.dumps(body)[:200])

    # Poll
    print(f"     polling GET every {POLL_INTERVAL}s (timeout {POLL_TIMEOUT}s)...")
    elapsed = 0
    while elapsed < POLL_TIMEOUT:
        time.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL
        status, body = request("GET",
            f"{base}/tickers/{ticker}/analysis?accession={accession}")
        print(f"     {elapsed:3d}s → HTTP {status}")
        if status == 200:
            ok(f"analysis ready after {elapsed}s")
            _validate_analysis_body(body)
            return
        if status != 202:
            fail(f"unexpected poll status {status}", json.dumps(body)[:200])

    fail(f"timed out after {POLL_TIMEOUT}s — worker may have crashed")


def test_second_post_returns_cached(base: str, ticker: str, accession: str) -> None:
    print(f"\n[7] POST again (same accession) — expect 200 cached")
    status, body = request("POST",
        f"{base}/tickers/{ticker}/analysis?accession={accession}",
        timeout=15)
    check("status 200", status == 200, f"got {status}: {body}")
    check("cached=True", body.get("cached") is True, json.dumps(body)[:200])
    ok("second POST hits cache as expected")


def _validate_analysis_body(body: dict) -> None:
    analysis = body.get("analysis", {})
    for field in ("accession", "ticker", "filing_date", "prediction",
                  "probability_up", "aspects"):
        check(f"  analysis.{field} present", field in analysis,
              json.dumps(analysis)[:200])
    aspects = analysis.get("aspects", {})
    for a in ("revenue", "cash_flow", "margins", "ebitda",
              "future_plans", "risk_factors", "guidance"):
        check(f"  aspects.{a} is float", isinstance(aspects.get(a), float),
              f"got {aspects.get(a)!r}")
    prob = analysis.get("probability_up")
    check("  probability_up in [0,1]",
          isinstance(prob, float) and 0.0 <= prob <= 1.0, str(prob))
    check("  prediction is up/down",
          analysis.get("prediction") in ("up", "down"),
          str(analysis.get("prediction")))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    base   = sys.argv[1].rstrip("/")
    ticker = sys.argv[2].upper() if len(sys.argv) > 2 else TICKER

    print(f"API base : {base}")
    print(f"Ticker   : {ticker}")

    accession = test_filings(base, ticker)
    test_filings_unknown_ticker(base)
    test_analysis_get_no_accession(base, ticker)
    test_analysis_get_bad_accession(base, ticker)
    test_analysis_post_no_accession(base, ticker)
    test_async_flow(base, ticker, accession)
    test_second_post_returns_cached(base, ticker, accession)

    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
