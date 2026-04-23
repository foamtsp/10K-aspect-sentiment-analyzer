#!/usr/bin/env python3
"""
Batch pre-fill: score the last N 10-K filings for each of the top-50 tickers.

Each ticker is processed oldest→newest so YoY deltas are available from the
second filing onward. Multiple tickers run in parallel (one thread each).

Usage:
    python scripts/batch_prefill.py <api-base-url> [--workers N] [--filings N] [--dry-run]

Examples:
    python scripts/batch_prefill.py https://xxxxx.execute-api.us-east-1.amazonaws.com
    python scripts/batch_prefill.py https://xxxxx... --workers 5 --filings 3
    python scripts/batch_prefill.py https://xxxxx... --dry-run   # show plan, no scoring
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import NamedTuple

# Top-50 tickers by market cap / trading volume (S&P 500 + Nasdaq heavyweights).
TOP_50 = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK-B", "JPM", "LLY",
    "V",    "UNH",  "XOM",  "MA",   "AVGO",  "PG",   "HD",   "COST",  "JNJ", "WMT",
    "ABBV", "CRM",  "BAC",  "MRK",  "CVX",   "NFLX", "KO",   "PEP",   "TMO", "ADBE",
    "ACN",  "LIN",  "MCD",  "AMD",  "DHR",   "TXN",  "CSCO", "WFC",   "PM",  "NEE",
    "INTU", "AMGN", "MS",   "RTX",  "HON",   "UPS",  "QCOM", "CAT",   "GS",  "IBM",
]

POLL_INTERVAL = 10   # seconds between cache polls
POLL_TIMEOUT  = 300  # give up on a single filing after 5 min


class FilingResult(NamedTuple):
    ticker: str
    accession: str
    filing_date: str
    status: str   # "cached" | "scored" | "failed" | "skipped"
    elapsed: float


def _request(method: str, url: str, timeout: int = 35) -> tuple[int, dict]:
    req = urllib.request.Request(url, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read())
        except Exception:
            body = {"error": str(e)}
        return e.code, body
    except Exception as e:
        return 0, {"error": str(e)}


def _cutoff_date(years_back: int) -> str:
    return (datetime.utcnow() - timedelta(days=365 * years_back)).strftime("%Y-%m-%d")


def score_filing(base: str, ticker: str, accession: str, filing_date: str) -> FilingResult:
    t0 = time.time()

    # POST to trigger (returns 200=cached, 202=started)
    status, body = _request("POST",
        f"{base}/tickers/{ticker}/analysis?accession={accession}")

    if status == 200:
        return FilingResult(ticker, accession, filing_date, "cached", time.time() - t0)

    if status != 202:
        return FilingResult(ticker, accession, filing_date,
                            f"failed(POST {status})", time.time() - t0)

    # Poll until cached
    elapsed = 0
    while elapsed < POLL_TIMEOUT:
        time.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL
        status, body = _request("GET",
            f"{base}/tickers/{ticker}/analysis?accession={accession}")
        if status == 200:
            return FilingResult(ticker, accession, filing_date, "scored", time.time() - t0)
        if status != 202:
            return FilingResult(ticker, accession, filing_date,
                                f"failed(poll {status})", time.time() - t0)

    return FilingResult(ticker, accession, filing_date,
                        f"timeout({POLL_TIMEOUT}s)", time.time() - t0)


def process_ticker(base: str, ticker: str, max_filings: int,
                   years_back: int, dry_run: bool) -> list[FilingResult]:
    results: list[FilingResult] = []
    cutoff = _cutoff_date(years_back)

    status, body = _request("GET", f"{base}/tickers/{ticker}/filings")
    if status != 200:
        print(f"  [{ticker}] ERROR fetching filings: {status} {body}")
        return results

    filings = body.get("filings", [])

    # Keep only filings within the lookback window, then take the N most recent.
    filings = [f for f in filings if f["filing_date"] >= cutoff]
    filings = filings[:max_filings]

    if not filings:
        print(f"  [{ticker}] no filings in last {years_back} years — skipping")
        return results

    # Reverse to oldest-first so each filing can compute YoY delta from the prior.
    filings = list(reversed(filings))

    print(f"  [{ticker}] {len(filings)} filings to score "
          f"({filings[0]['filing_date']} → {filings[-1]['filing_date']})")

    for f in filings:
        if dry_run:
            print(f"    [dry-run] {ticker} {f['filing_date']} {f['accession']}")
            results.append(FilingResult(ticker, f["accession"], f["filing_date"],
                                        "skipped", 0.0))
            continue

        result = score_filing(base, ticker, f["accession"], f["filing_date"])
        icon = "✓" if result.status in ("cached", "scored") else "✗"
        print(f"    {icon} {ticker} {result.filing_date}  "
              f"{result.status}  ({result.elapsed:.0f}s)")
        results.append(result)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("api_base", help="API Gateway base URL")
    parser.add_argument("--workers",  type=int, default=5,
                        help="Parallel tickers (default 5)")
    parser.add_argument("--filings",  type=int, default=5,
                        help="Most recent 10-Ks per ticker (default 5 = ~5 years)")
    parser.add_argument("--years",    type=int, default=6,
                        help="Only include filings within this many years (default 6)")
    parser.add_argument("--tickers",  nargs="+",
                        help="Override ticker list (default: top-50)")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Show plan without scoring anything")
    args = parser.parse_args()

    base    = args.api_base.rstrip("/")
    tickers = [t.upper() for t in args.tickers] if args.tickers else TOP_50

    print(f"API        : {base}")
    print(f"Tickers    : {len(tickers)}")
    print(f"Filings    : up to {args.filings} per ticker (last {args.years} years)")
    print(f"Workers    : {args.workers} parallel tickers")
    print(f"Dry run    : {args.dry_run}")
    print(f"Est. time  : ~{len(tickers) * args.filings * 25 // args.workers // 60} min "
          f"(first-time, ignores cache hits)\n")

    all_results: list[FilingResult] = []
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_ticker, base, t, args.filings,
                        args.years, args.dry_run): t
            for t in tickers
        }
        for fut in as_completed(futures):
            ticker = futures[fut]
            try:
                all_results.extend(fut.result())
            except Exception as exc:
                print(f"  [{ticker}] EXCEPTION: {exc}")

    # Summary
    total   = len(all_results)
    cached  = sum(1 for r in all_results if r.status == "cached")
    scored  = sum(1 for r in all_results if r.status == "scored")
    failed  = [r for r in all_results if r.status not in ("cached", "scored", "skipped")]
    elapsed = time.time() - t_start

    print(f"\n{'='*55}")
    print(f"Done in {elapsed:.0f}s")
    print(f"  Total filings : {total}")
    print(f"  Already cached: {cached}")
    print(f"  Freshly scored: {scored}")
    print(f"  Failed        : {len(failed)}")
    if failed:
        print("\nFailed filings:")
        for r in failed:
            print(f"  {r.ticker:8s} {r.filing_date}  {r.accession}  {r.status}")


if __name__ == "__main__":
    main()
