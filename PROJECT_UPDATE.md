# Project Update — 10-K Sentiment → Price Direction Pipeline

## 1. AWS Deployment Status

Our system is deployed on AWS as a **pull-based, serverless pipeline** driven by the user's UI input. The dashboard takes a ticker, the API checks SEC EDGAR for the latest 10-K, and either returns cached analysis or kicks off async scoring and polls until ready. There is no scheduled ingestion and no watchlist — everything is on-demand.

### Services in use

| Layer | AWS service | Role |
|---|---|---|
| UI | **ECS Fargate** (Streamlit container) | Dashboard on port 8501, behind an ALB |
| Entry point | **API Gateway** (HTTP API v2) | `GET /tickers/{t}/analysis`, `POST /tickers/{t}/refresh`, `GET /jobs/{id}` |
| Compute | **Lambda** × 5 | `api`, `ingest`, `text_extract`, `predict`, `trading_signal` |
| Storage (raw) | **S3** | `raw/` (HTML from EDGAR), `parsed/` (sentence-segmented JSON), `models/xgb/v1/` (trained artifacts) |
| State | **Aurora PostgreSQL Serverless v2** | `filings`, `aspect_scores`, `predictions`, `trading_signals`, `realized_outcomes`, `jobs` |
| ML inference | **SageMaker** real-time endpoints | `finbert-endpoint` (ProsusAI/finbert, HuggingFace container), `xgboost-endpoint` (our trained booster + isotonic calibrator) |
| Fan-out | **SNS → SQS FIFO** | Prediction topic → trading-signal Lambda → execution queue |
| Secrets / config | **Secrets Manager** | DB credentials |
| IaC | **CloudFormation** | 31 resources, 7 outputs, one template |

### How the pieces fit together

```
  Streamlit (ECS Fargate)
        │  GET /tickers/{t}/analysis
        ▼
  API Gateway ──► api Lambda ──┬─► EDGAR submissions.json (is newer 10-K available?)
                               ├─► Aurora (return cached analysis if fresh)
                               └─► ingest Lambda (async, InvocationType=Event)
                                        ▼
                                   S3 raw/ ──► text_extract Lambda ──► S3 parsed/
                                                                            ▼
                                                                      predict Lambda
                                                                        ├─► SageMaker FinBERT
                                                                        ├─► SageMaker XGBoost
                                                                        ├─► Aurora (jobs → ready)
                                                                        └─► SNS ──► trading_signal ──► SQS
```

### Deployment status

| Component | Status |
|---|---|
| CloudFormation stack (VPC, Aurora, S3, Lambdas, API Gateway, ECS, SNS, SQS, IAM) | **Deployed** via `scripts/deploy.sh` |
| Aurora schema (6 tables incl. `jobs`) | **Deployed** via `psql -f sql/schema.sql` |
| Streamlit dashboard image (ECR + ECS service) | **Deployed** |
| XGBoost SageMaker endpoint | **Artifacts trained; endpoint deploy scripted** — `scripts/deploy_model.sh v1` uploads `booster.json` / `calibrator.pkl` / `feature_cols.json` / `model.tar.gz` from local `models/` to S3 and creates the endpoint. |
| FinBERT SageMaker endpoint | **Remaining** — pending deploy; will use the HuggingFace SageMaker container pointing at `ProsusAI/finbert`. |

### Remaining work before demo

1. Stand up the FinBERT endpoint (one `HuggingFaceModel(...).deploy()` call — container is already in `sagemaker/finbert/`).
2. End-to-end smoke test: enter AAPL in the dashboard → verify a fresh 10-K is scored and the radar chart renders.
3. Realized-outcome backfill Lambda (nightly job to populate `realized_outcomes` once the 30-day horizon elapses) — a stretch item, not required for the demo.

---

## 2. Final Model Selection

### Selected model

**Calibrated XGBoost** classifier predicting 30-day price direction (`label_30d ∈ {up, down}`) from a 23-dim feature vector of aspect-level 10-K sentiment:

- **Features per filing (23):** For each of 7 aspects (revenue, cash flow, margins, EBITDA, future plans, risk factors, guidance) we compute `{aspect}_score` (mean signed FinBERT confidence on tagged sentences), `{aspect}_count` (sentence count), and `{aspect}_delta` (**YoY change vs. the same ticker's prior 10-K**). Plus `sic` industry code and `n_sentences`.
- **Head:** `XGBClassifier` (booster.json, 54 KB) → `CalibratedClassifierCV(method="isotonic", cv="prefit")` trained on the validation split. This is what the `0.70` confidence gate in `trading_signal` actually observes — calibrated probabilities, not raw logits.

### Why XGBoost + calibration over the Memo 2 alternatives

| Candidate from Memo 2 | Outcome | Why not selected |
|---|---|---|
| Logistic regression on aspect scores | Weakest ranking; underfit the non-linear interactions between margins and guidance sentiment | Simple, interpretable, but left ~5 AUC points on the table |
| FinBERT end-to-end (no XGBoost head) | Strong sentence-level sentiment but no mechanism to weight aspects, use YoY deltas, or incorporate SIC | Sentiment ≠ directional forecast; signal is diluted at filing level |
| Random forest | Comparable raw AUC to XGBoost, slower inference, no native support for our preferred calibration pipeline | Close second — kept only as a sanity check |
| LSTM / transformer on raw sentence streams | Too little data (213 filings) to train end-to-end; risk of overfit and high inference latency | Impractical given dataset size |

XGBoost won on three axes:

1. **Ranking performance** — highest AUC on the held-out test split (see below).
2. **Calibrated probabilities** — isotonic regression on the validation split gives reliable confidence scores, which is exactly what the trading-signal gate needs.
3. **Latency and cost** — the booster fits on an `ml.t2.medium` SageMaker endpoint (~$30/mo) and responds in <50 ms. FinBERT is the expensive call (~1 s per batch of 32 sentences, ~15–20 s per filing).

### Final test-set performance

Model version **v1**, trained on the full 10-K aspect-scored corpus with a time-ordered split (188 train / 12 val / 13 test filings).

| Metric | Validation | Test |
|---|---|---|
| AUC (raw XGBoost) | 0.653 | 0.750 |
| **AUC (after isotonic calibration)** | **0.778** | **0.722** |

Calibration slightly reduces raw AUC on the test split (0.75 → 0.72) but buys us the reliability curve the downstream signal gate depends on. Directional accuracy at the 0.70 calibrated-probability threshold corresponds to ~70% observed hit rate rather than a raw logit percentile — we traded ~3 AUC points for a threshold with real operational meaning.

### Tradeoffs we accepted

- **Small test set (n = 13 filings)** — the confidence interval on 0.72 AUC is wide. We treat this as a plausibility check, not a promise; the demo narrative is methodology + end-to-end plumbing, not a production trading claim.
- **Interpretability** — XGBoost gain-based feature importance is reported in the training notebook, but we lose the per-coefficient story logistic regression would give. Mitigated by the dashboard's radar chart, which shows the underlying aspect vector.
- **Cold-start compute** — First-time scoring of a new 10-K runs FinBERT over ~2–5k sentences and takes 15–25 s. The UI handles this with async jobs + polling rather than forcing the user to wait synchronously.
- **No realized-outcome loop yet** — we cannot show a live hit-rate from production; only the held-out test metric.

---

## 3. User Interface Plan

### End user

The primary user is a **retail/discretionary equity analyst or student investor** evaluating whether a company's just-filed 10-K points to upside or downside risk over the next month. They are comfortable with tickers and basic financial vocabulary but are not running their own ML infrastructure.

### Platform

**Streamlit** app, containerized with Docker and running on ECS Fargate behind an ALB. Chosen over Tableau/web-form alternatives because:

- We need a live, stateful polling loop (`GET /jobs/{id}` every 3 s while a filing is being scored) — trivial in Streamlit, awkward in Tableau.
- We wanted the UI and the inference pipeline in the same repo and deploy path.
- No auth / no embedding in an external portal required at this stage.

### Input / Output contract

- **Input:** a ticker string (e.g. `AAPL`), plus two buttons: **Analyze** (use cached if fresh) and **Force refresh** (re-score even if cached).
- **Output** (from `GET /tickers/{t}/analysis`):
  - Filing date, accession number, 30-day prediction (`up` / `down`), calibrated confidence (e.g. `74%`), model version, `is_latest` flag.
  - 7-aspect sentiment radar chart with scores bounded in [-1, 1].
  - Tabular aspect scores with a green/red/white marker for >0.1 / <-0.1 / otherwise.
- **Async UX:** If EDGAR has a newer 10-K than the cache, the API returns the stale analysis immediately **plus** a `job_id`. The dashboard shows the stale result, displays a "Scoring new 10-K… (Xs elapsed)" banner, polls `/jobs/{id}` every 3 s (timeout 120 s), and re-renders when `status=ready`.

### Screens

The dashboard is a single screen with a sidebar (ticker input + two buttons) and a main pane containing four top-row metrics (Filing date / Prediction / Confidence / Model version) and two side-by-side panels (radar chart / aspect score table). Early working prototype is live in `dashboard/app.py` and builds locally with:

```bash
export API_BASE_URL=http://<api>
streamlit run dashboard/app.py
```

Screenshots of the running prototype will be attached to the final submission.

### Error states handled

- Ticker not on EDGAR → 404 surfaces as a friendly error.
- Job timeout (>120 s) → banner telling the user it will finish in the background.
- No 10-K ever scored for this ticker → explicit "No analysis yet" message.
- API unreachable → exception caught, user sees status + code.
