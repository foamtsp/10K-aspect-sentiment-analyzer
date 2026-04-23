"""Aggregate sentence-level FinBERT output into a per-filing aspect vector.

A filing yields ~hundreds to thousands of sentences, each tagged with zero or
more aspects (see aspects.tag). We collapse those into one number per aspect
per filing by taking the signed-confidence mean:

    sentiment_score = P(positive) - P(negative)

Missing aspects (no matching sentence) get 0 so the downstream classifier has
a fixed-width feature vector regardless of filing length. We also expose
counts so the model can weight "aspect discussed heavily" differently from
"aspect barely mentioned".
"""
from __future__ import annotations

from dataclasses import dataclass

from .aspects import ASPECTS, tag


@dataclass
class AspectVector:
    scores: dict[str, float]
    counts: dict[str, int]

    def as_feature_row(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for a in ASPECTS:
            out[f"{a}_score"] = self.scores.get(a, 0.0)
            out[f"{a}_count"] = float(self.counts.get(a, 0))
        return out


def aggregate(
    sentences: list[str],
    sentiments: list[dict[str, float]],
    sections: list[str] | None = None,
) -> AspectVector:
    """Build a per-aspect sentiment vector for one filing.

    `sentiments[i]` must have keys "positive", "negative", "neutral" for
    sentence `sentences[i]`. Sentences can contribute to multiple aspects.
    """
    if len(sentences) != len(sentiments):
        raise ValueError("sentences and sentiments length mismatch")
    sections = sections or [None] * len(sentences)

    sums: dict[str, float] = {a: 0.0 for a in ASPECTS}
    counts: dict[str, int] = {a: 0 for a in ASPECTS}

    for sent, sentiment, section in zip(sentences, sentiments, sections):
        signed = sentiment.get("positive", 0.0) - sentiment.get("negative", 0.0)
        for match in tag(sent, section):
            sums[match.aspect] += signed
            counts[match.aspect] += 1

    scores = {a: (sums[a] / counts[a]) if counts[a] else 0.0 for a in ASPECTS}
    return AspectVector(scores=scores, counts=counts)
