"""EDGAR helpers: ticker→CIK lookup, list 10-Ks, fetch primary document."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import urllib3

USER_AGENT = os.environ.get("SEC_USER_AGENT", "research-bot contact@example.com")
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{primary_doc}"

_http = urllib3.PoolManager(headers={"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"})
_TICKER_MAP: dict[str, int] | None = None


@dataclass
class FilingRef:
    accession: str
    filing_date: str
    form: str
    primary_doc: str


def cik_for(ticker: str) -> int | None:
    """Resolve ticker → CIK via EDGAR's master map (cached per container)."""
    global _TICKER_MAP
    if _TICKER_MAP is None:
        resp = _http.request("GET", TICKERS_URL)
        if resp.status != 200:
            raise RuntimeError(f"ticker map fetch failed: {resp.status}")
        data = json.loads(resp.data.decode())
        _TICKER_MAP = {row["ticker"].upper(): int(row["cik_str"]) for row in data.values()}
    return _TICKER_MAP.get(ticker.upper())


def list_10ks(cik: int, limit: int = 10) -> list[FilingRef]:
    """Return the most recent 10-K / 10-K/A filings for a CIK."""
    resp = _http.request("GET", SUBMISSIONS_URL.format(cik=cik))
    if resp.status != 200:
        return []
    recent = json.loads(resp.data.decode()).get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    out: list[FilingRef] = []
    for i, form in enumerate(forms):
        if form in ("10-K", "10-K/A"):
            out.append(FilingRef(
                accession=recent["accessionNumber"][i],
                filing_date=recent["filingDate"][i],
                form=form,
                primary_doc=recent["primaryDocument"][i],
            ))
            if len(out) >= limit:
                break
    return out


def fetch_filing_html(cik: int, accession: str, primary_doc: str) -> bytes:
    """Download the primary HTML document for a given filing."""
    acc_nodash = accession.replace("-", "")
    url = ARCHIVE_URL.format(cik=cik, acc_nodash=acc_nodash, primary_doc=primary_doc)
    resp = _http.request("GET", url)
    if resp.status != 200:
        raise RuntimeError(f"EDGAR fetch failed ({resp.status}): {url}")
    return resp.data
