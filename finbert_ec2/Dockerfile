FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.2.2 \
 && pip install --no-cache-dir \
    transformers==4.41.2 fastapi==0.111.0 uvicorn==0.30.1 pydantic==2.7.1

ENV HF_HOME=/opt/finbert/hf

# Pre-bake model weights into the image so first request is warm.
RUN python - <<'PY'
from transformers import AutoModelForSequenceClassification, AutoTokenizer
AutoTokenizer.from_pretrained("ProsusAI/finbert")
AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
PY

COPY finbert_ec2/server.py /app/server.py

EXPOSE 8080

CMD ["python", "server.py"]
