"""FinBERT HTTP server — drop-in replacement for the SageMaker endpoint.

Keeps the exact same JSON contract as `sagemaker/finbert/inference.py`:

    POST /predict   Body:   {"sentences": ["…", "…"]}
                    Reply:  {"predictions": [{"positive": .., "negative": .., "neutral": ..}, ...]}
    GET  /health    Reply:  {"status": "ok", "device": "cpu"|"cuda"}

Runs on an EC2 instance (see infrastructure/cloudformation.yaml for how it's
launched). The scorer Lambda reaches it via its VPC-internal private IP on
port 8080.
"""
from __future__ import annotations

import os
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = os.environ.get("FINBERT_MODEL_ID", "ProsusAI/finbert")
LABELS = ("positive", "negative", "neutral")
MAX_LEN = 256
BATCH_SIZE = 32
PORT = int(os.environ.get("PORT", "8080"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).to(device).eval()

app = FastAPI()


class PredictRequest(BaseModel):
    sentences: list[str]


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "device": str(device), "model": MODEL_ID}


@app.post("/predict")
def predict(req: PredictRequest) -> dict[str, list[dict[str, float]]]:
    sents = [s for s in req.sentences if isinstance(s, str) and s.strip()]
    results: list[dict[str, float]] = []
    with torch.no_grad():
        for i in range(0, len(sents), BATCH_SIZE):
            chunk = sents[i : i + BATCH_SIZE]
            enc = tokenizer(
                chunk, padding=True, truncation=True,
                max_length=MAX_LEN, return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            probs = torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()
            for p in probs:
                results.append({lab: float(p[j]) for j, lab in enumerate(LABELS)})
    return {"predictions": results}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
