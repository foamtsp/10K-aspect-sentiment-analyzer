"""Aspect taxonomy and sentence-to-aspect classifier.

The pipeline needs to turn a 10-K sentence into one or more of the business
aspects we track (revenue, cash_flow, margins, ebitda, future_plans,
risk_factors, guidance). FinBERT gives us sentiment per sentence; this module
answers "which aspect(s) does this sentence talk about?" so we can build a
per-aspect sentiment vector per filing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

ASPECTS = (
    "revenue",
    "cash_flow",
    "margins",
    "ebitda",
    "future_plans",
    "risk_factors",
    "guidance",
)

_KEYWORDS: dict[str, tuple[str, ...]] = {
    "revenue": (
        "revenue", "revenues", "net sales", "total sales", "top line",
        "net revenue", "product sales", "service revenue", "billings",
    ),
    "cash_flow": (
        "cash flow", "cash flows", "operating cash", "free cash flow",
        "cash from operations", "cash and cash equivalents", "working capital",
        "liquidity",
    ),
    "margins": (
        "gross margin", "operating margin", "net margin", "profit margin",
        "margin expansion", "margin compression", "cost of goods sold",
        "cogs", "gross profit",
    ),
    "ebitda": (
        "ebitda", "adjusted ebitda", "earnings before interest",
        "operating income", "operating profit", "operating earnings",
    ),
    "future_plans": (
        "we plan", "we intend", "we expect to", "we will", "strategy",
        "initiative", "roadmap", "expansion", "invest in", "planned",
        "in the coming year", "next fiscal", "long-term", "future growth",
    ),
    "risk_factors": (
        "risk", "risks", "uncertainty", "adverse", "material adverse",
        "could harm", "may fail", "litigation", "regulatory", "volatility",
        "downturn", "recession", "dependence on", "could impact",
    ),
    "guidance": (
        "guidance", "outlook", "forecast", "projected", "anticipate",
        "estimate", "target", "we believe", "expected to be", "forward-looking",
    ),
}

_PATTERNS: dict[str, re.Pattern[str]] = {
    aspect: re.compile(r"\b(" + "|".join(re.escape(k) for k in kws) + r")\b", re.IGNORECASE)
    for aspect, kws in _KEYWORDS.items()
}

_SECTION_PRIORS: dict[str, tuple[str, ...]] = {
    "section_1": ("revenue", "future_plans"),
    "section_1A": ("risk_factors",),
    "section_7": ("revenue", "margins", "ebitda", "cash_flow", "guidance", "future_plans"),
    "section_7A": ("risk_factors",),
}


@dataclass(frozen=True)
class AspectMatch:
    aspect: str
    matched_terms: tuple[str, ...]


def tag(sentence: str, section: str | None = None) -> list[AspectMatch]:
    """Return all aspects that appear in the sentence.

    A sentence can map to multiple aspects (e.g. "revenue margins compressed
    due to rising input costs" → revenue + margins). If no keyword matches but
    the section has a strong prior, fall back to the section prior with a
    single empty-term match.
    """
    out: list[AspectMatch] = []
    for aspect, pat in _PATTERNS.items():
        hits = tuple(m.group(0).lower() for m in pat.finditer(sentence))
        if hits:
            out.append(AspectMatch(aspect, hits))
    if not out and section in _SECTION_PRIORS:
        out = [AspectMatch(a, ()) for a in _SECTION_PRIORS[section]]
    return out


def tag_batch(sentences: list[str], sections: list[str] | None = None) -> list[list[AspectMatch]]:
    sections = sections or [None] * len(sentences)
    return [tag(s, sec) for s, sec in zip(sentences, sections)]
