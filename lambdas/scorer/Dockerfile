# Amazon Linux 2023 base (glibc 2.34) — manylinux_2_28 wheels install directly.
FROM public.ecr.aws/lambda/python:3.12

# Pin numpy first so torch, xgboost, and scikit-learn all agree on the version.
RUN pip install --no-cache-dir "numpy==1.26.4"

# CPU-only PyTorch. --extra-index-url keeps PyPI accessible for other deps.
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    "torch==2.2.2"

COPY lambdas/scorer/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir \
    "transformers==4.41.2" \
    -r /tmp/requirements.txt

# Pre-bake FinBERT weights so the first inference call is warm.
ENV HF_HOME=/opt/finbert/hf
RUN python -c "\
from transformers import AutoModelForSequenceClassification, AutoTokenizer; \
AutoTokenizer.from_pretrained('JanhaviS14/finance-sentiment-mini-finbert'); \
AutoModelForSequenceClassification.from_pretrained('JanhaviS14/finance-sentiment-mini-finbert')"

COPY common/ ${LAMBDA_TASK_ROOT}/common/
COPY lambdas/scorer/handler.py ${LAMBDA_TASK_ROOT}/handler.py

COPY models/booster.json        /opt/model/booster.json
COPY models/calibrator.pkl      /opt/model/calibrator.pkl
COPY models/feature_cols.json   /opt/model/feature_cols.json

CMD ["handler.handler"]
