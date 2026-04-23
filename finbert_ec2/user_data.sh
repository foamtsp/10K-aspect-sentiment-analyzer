#!/bin/bash
# Bootstrap script for the FinBERT EC2 instance.
# Launched by CloudFormation via the FinBertInstance's UserData (this file
# is the readable reference; the template inlines the same logic).
#
# Target AMI: Ubuntu 22.04 (plain, no DLAMI dependency — keeps it portable).
# Target instance: m5.xlarge (CPU) by default; override to g4dn.xlarge for GPU.
# Full bootstrap takes ~3-5 min on a fresh instance (torch CPU wheel + model dl).
set -euo pipefail
exec > /var/log/finbert-bootstrap.log 2>&1

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3.10 python3.10-venv python3-pip

mkdir -p /opt/finbert
python3 -m venv /opt/finbert/venv
/opt/finbert/venv/bin/pip install --upgrade pip wheel

# CPU-only torch wheel (~200 MB) — much smaller than default GPU wheel.
# If the instance has a GPU, swap this to the CUDA wheel manually.
/opt/finbert/venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu \
    "torch==2.2.2"
/opt/finbert/venv/bin/pip install \
    "transformers==4.41.2" "fastapi==0.111.0" "uvicorn==0.30.1" "pydantic==2.7.1"

# Pre-download FinBERT so the server is warm on first request.
HF_HOME=/opt/finbert/hf /opt/finbert/venv/bin/python - <<'PY'
from transformers import AutoModelForSequenceClassification, AutoTokenizer
AutoTokenizer.from_pretrained("ProsusAI/finbert")
AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
PY

cat > /opt/finbert/server.py <<'PY'
# --- copy of finbert_ec2/server.py (keep in sync) ---
import os
import torch, uvicorn
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
def health():
    return {"status": "ok", "device": str(device), "model": MODEL_ID}

@app.post("/predict")
def predict(req: PredictRequest):
    sents = [s for s in req.sentences if isinstance(s, str) and s.strip()]
    results = []
    with torch.no_grad():
        for i in range(0, len(sents), BATCH_SIZE):
            chunk = sents[i:i+BATCH_SIZE]
            enc = tokenizer(chunk, padding=True, truncation=True,
                            max_length=MAX_LEN, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            probs = torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()
            for p in probs:
                results.append({lab: float(p[j]) for j, lab in enumerate(LABELS)})
    return {"predictions": results}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
PY

cat > /etc/systemd/system/finbert.service <<'UNIT'
[Unit]
Description=FinBERT HTTP server
After=network.target

[Service]
ExecStart=/opt/finbert/venv/bin/python /opt/finbert/server.py
WorkingDirectory=/opt/finbert
Environment=HF_HOME=/opt/finbert/hf
Restart=always
User=root

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable finbert
systemctl start finbert
