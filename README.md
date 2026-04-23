# 10-K Sentiment → Price Direction Pipeline (MVP)

Pull-based pipeline. User types a ticker in the Streamlit dashboard, picks
a 10-K from the dropdown, and the system returns aspect sentiment plus a
30-day price-direction prediction. First-time scoring takes ~20–30 s (cold
start + inference); subsequent reads are instant (MySQL cache). The
dashboard polls every 5 s while scoring is in progress.

## Architecture

```
Streamlit (ECS Fargate, public)
      │  GET  /tickers/{t}/filings
      │  POST /tickers/{t}/analysis?accession=...   ← trigger async scoring
      │  GET  /tickers/{t}/analysis?accession=...   ← poll until 200
      ▼
API Gateway (HTTP API)
      ▼
scorer Lambda (container image, no VPC)
      │  trigger path (POST, ≤1 s):
      │   ├─ MySQL lookup → 200 cached  (cache hit)
      │   └─ self-invoke InvocationType=Event → 202 pending  (cache miss)
      │
      │  worker path (_worker=true, async):
      │   ├─ EDGAR          (ticker → CIK, list 10-Ks, fetch HTML)
      │   ├─ FinBERT        (bundled in image; CPU inference, no external call)
      │   ├─ XGBoost+isotonic (bundled in image)
      │   └─ MySQL upsert   (aspect scores + YoY deltas + prediction)
      ▼
Existing churn-db RDS (MySQL 8, port 3306 open to 0.0.0.0/0)
```

The scorer Lambda is split into two execution paths to work around API
Gateway's 29-second timeout. The trigger path returns in < 1 s; the heavy
work (EDGAR fetch + FinBERT inference + XGBoost) runs in a self-invoked
async worker path. The dashboard polls `GET /analysis` every 5 s until the
result appears in the MySQL cache.

FinBERT runs **inside the scorer Lambda** — the model weights (~440 MB)
and PyTorch CPU wheel (~200 MB) are baked into the scorer container image
at build time. No separate service, no networking. The Lambda is allocated
8 GB memory (≈ 4 vCPUs). This avoids both SageMaker quota limits and the
`ec2:RunInstances` / `servicediscovery:CreatePrivateDnsNamespace` deny
policies common in Learner-Lab accounts.

## Repository layout

```
common/                 aspects.py, sections.py, aggregation.py, db.py, edgar.py
lambdas/scorer/         Scorer Lambda (container image): Dockerfile, handler.py
dashboard/              Streamlit app (calls API), Dockerfile
finbert_ec2/            Reference only — standalone FinBERT FastAPI server (not deployed)
models/                 Trained XGBoost booster + isotonic calibrator (bundled in scorer image)
infrastructure/         cloudformation.yaml
sql/schema.sql          Single-table MySQL schema (scored_filings + delta columns)
scripts/
  deploy.sh             Build, push, apply schema, deploy CloudFormation
  test_api.py           End-to-end API test suite (7 tests)
  batch_prefill.py      Bulk pre-score top-50 tickers × last 5 years (parallel)
  dashboard_url.sh      Print the ECS task's public IP / dashboard URL
notebooks/              Training + analysis notebooks
```

## REST API

| Method | Route | Description |
|--------|-------|-------------|
| `GET`  | `/tickers/{ticker}/filings` | Recent 10-Ks from EDGAR (for the UI dropdown). |
| `POST` | `/tickers/{ticker}/analysis?accession=...` | Trigger scoring. Returns `200` (cached) or `202` (scoring started). |
| `POST` | `/tickers/{ticker}/analysis?accession=...&refresh=1` | Force re-score, overwriting the cache. |
| `GET`  | `/tickers/{ticker}/analysis?accession=...` | Poll for result. Returns `200` when ready, `202` while pending. |

Response shape for `/analysis` (status 200):

```json
{
  "cached": false,
  "analysis": {
    "accession": "0000320193-24-000123",
    "ticker": "AAPL",
    "cik": "320193",
    "filing_date": "2024-10-31",
    "form": "10-K",
    "prediction": "up",
    "probability_up": 0.74,
    "horizon_days": 30,
    "model_version": "v1",
    "n_sentences": 4217,
    "aspects": {
      "revenue": 0.41, "cash_flow": 0.22, "margins": -0.08,
      "ebitda": 0.11, "future_plans": 0.33,
      "risk_factors": -0.17, "guidance": 0.05
    },
    "deltas": {
      "revenue": 0.12, "cash_flow": null, "margins": -0.03,
      "ebitda": 0.05, "future_plans": 0.08,
      "risk_factors": -0.04, "guidance": null
    }
  }
}
```

`deltas` values are `null` when no prior filing exists in the cache for
that ticker (i.e., the first scored filing has no YoY reference).

Pending response (status 202):

```json
{ "status": "pending" }
```

## Deploy

> **No AWS CLI keys locally?** See [DEPLOY_CONSOLE.md](DEPLOY_CONSOLE.md) —
> same commands, run from AWS CloudShell in the browser.

```bash
# 1. (One-off) Train the model and drop artifacts into ./models/:
#    booster.json, calibrator.pkl, feature_cols.json, metrics.json

# 2. Deploy the app stack (builds + pushes both container images,
#    applies schema, deploys CloudFormation, and rolls ECS)
scripts/deploy.sh <vpc-id> <public-subnets-csv> \
    <db-host> <db-username> <db-password> [db-name]

# 3. Open TCP 3306 on churn-db's security group to 0.0.0.0/0
#    (the scorer Lambda runs outside the VPC to reach EDGAR directly)

# 4. Smoke-test:
curl "https://<api-id>.execute-api.us-east-1.amazonaws.com/tickers/AAPL/filings"

# 5. Get the dashboard URL:
bash scripts/dashboard_url.sh

# 6. (Optional) Pre-score top-50 tickers × last 5 years:
python scripts/batch_prefill.py https://<api-id>.execute-api.us-east-1.amazonaws.com
```

## Local development

Run the dashboard against a deployed API:

```bash
export API_BASE_URL=https://<api-id>.execute-api.us-east-1.amazonaws.com
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```

## Key design notes

- **Async Lambda self-invocation.** Scoring a cold filing takes ~20–30 s
  (EDGAR fetch + FinBERT inference), which exceeds API Gateway's 29-second
  hard limit. The trigger path returns 202 immediately and re-invokes itself
  with `InvocationType=Event`; the dashboard polls `GET /analysis` every 5 s
  until the result is in MySQL. No SNS, SQS, or Step Functions needed.
- **HTML, not OCR.** 10-Ks are structured HTML on EDGAR. `common/sections.py`
  parses and segments Item 1 / 1A / 7 / 7A — cheaper and more accurate
  than Textract.
- **Tag-then-score.** `common/aspects.py` regex-tags sentences per aspect
  *before* calling FinBERT. Sentences with zero matches are dropped —
  typically 5–8× fewer sentences per invocation.
- **FinBERT bundled in the scorer Lambda.** Learner-Lab accounts deny both
  `ec2:RunInstances` and `servicediscovery:CreatePrivateDnsNamespace`.
  Bundling the model weights + PyTorch CPU into the scorer image removes
  every external dependency for inference: no EC2, no Fargate FinBERT
  service, no Cloud Map. The scorer image is ~1.5 GB; Lambda cold start
  takes ~20–30 s (model load); warm invocations are fast.
  Model: `JanhaviS14/finance-sentiment-mini-finbert` (128-token max for speed).
- **Lambda outside VPC.** Removing `VpcConfig` lets the Lambda reach EDGAR
  directly over the internet without a NAT Gateway (which Learner Lab also
  blocks). RDS port 3306 is opened to `0.0.0.0/0` — acceptable for a demo.
- **Inline XGBoost.** The calibrated booster is ~55 KB; baking it into the
  scorer image removes a whole moving part and keeps latency predictable.
- **YoY deltas.** At inference time, the worker looks up the same ticker's
  prior 10-K from the cache table to compute filing-over-filing score deltas.
  The `batch_prefill.py` script processes filings oldest→newest so each
  filing can reference the previous one already in the cache.
- **Calibrated probabilities.** Isotonic regression on the validation split;
  displayed `probability_up` is a true frequency estimate, not a raw logit.
- **Cache = source of truth for YoY.** The `scored_filings` table serves
  both as the prediction cache (lookup by `accession`) and the YoY-delta
  source (lookup by `ticker` + `filing_date < x`). One table; no FKs.

## Open items

- No auth on the API (wide-open CORS). For production, add API Gateway
  usage plans or a Cognito authorizer.
- No ALB in front of the ECS service. The dashboard is reached via the
  task's public IP — fine for a demo, not for a stable URL.
- No realized-outcome backfill. A nightly Lambda to populate actual 30-day
  returns would enable a live hit-rate display, but it's out of scope for
  the MVP.
