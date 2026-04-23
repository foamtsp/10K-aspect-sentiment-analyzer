"""Parse a 10-K HTML filing into section-segmented plain text.

SEC 10-K filings are structured HTML, not scanned PDFs. We skip OCR entirely
and parse the sections we actually use downstream:

    - section_1   Business
    - section_1A  Risk Factors
    - section_7   MD&A
    - section_7A  Quantitative and Qualitative Disclosures About Market Risk
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from bs4 import BeautifulSoup

SECTION_ANCHORS: dict[str, re.Pattern[str]] = {
    "section_1": re.compile(r"item\s*1\b(?!\s*[aA0-9])", re.IGNORECASE),
    "section_1A": re.compile(r"item\s*1\s*a\b", re.IGNORECASE),
    "section_7": re.compile(r"item\s*7\b(?!\s*[aA0-9])", re.IGNORECASE),
    "section_7A": re.compile(r"item\s*7\s*a\b", re.IGNORECASE),
}

_ORDER = ("section_1", "section_1A", "section_7", "section_7A")

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'])")


@dataclass
class ParsedFiling:
    cik: str
    ticker: str
    filing_date: str
    accession: str
    sections: dict[str, list[str]]  # section -> sentences


def html_to_text(html: bytes | str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "table"]):
        tag.decompose()
    text = soup.get_text(" ")
    return re.sub(r"\s+", " ", text).strip()


def split_into_sections(text: str) -> dict[str, str]:
    """Split a 10-K body into the four target sections by anchor regex.

    Approach: find the first occurrence of each section anchor, then slice
    between consecutive anchors in document order. The EDGAR filings repeat
    "Item 1" inside the Table of Contents, so we pick the second occurrence
    of each anchor when available (the first being the TOC entry).
    """
    spans: dict[str, int] = {}
    for name, pat in SECTION_ANCHORS.items():
        hits = [m.start() for m in pat.finditer(text)]
        if len(hits) >= 2:
            spans[name] = hits[1]
        elif hits:
            spans[name] = hits[0]
    ordered = sorted(((n, p) for n, p in spans.items()), key=lambda t: t[1])
    out: dict[str, str] = {}
    for i, (name, start) in enumerate(ordered):
        end = ordered[i + 1][1] if i + 1 < len(ordered) else len(text)
        out[name] = text[start:end]
    return out


def sentences(section_text: str, min_len: int = 20, max_len: int = 2000) -> list[str]:
    raw = _SENT_SPLIT.split(section_text)
    out = []
    for s in raw:
        s = s.strip()
        if min_len <= len(s) <= max_len:
            out.append(s)
    return out


def parse_filing(
    html: bytes | str,
    *,
    cik: str,
    ticker: str,
    filing_date: str,
    accession: str,
) -> ParsedFiling:
    text = html_to_text(html)
    by_section = split_into_sections(text)
    return ParsedFiling(
        cik=cik,
        ticker=ticker,
        filing_date=filing_date,
        accession=accession,
        sections={name: sentences(body) for name, body in by_section.items() if name in _ORDER},
    )
