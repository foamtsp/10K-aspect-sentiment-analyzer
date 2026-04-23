# Deploying from the AWS Console (no local CLI required)

If you don't have AWS access keys on your laptop, use **AWS CloudShell** — a
browser-based terminal inside the AWS Console with your credentials already
wired up. It has `aws`, `git`, `docker`, `python3`, `pip`, `mysql`, and `zip`
pre-installed. Everything below runs inside CloudShell unless marked
**[Console]** (meaning: click through the web UI).

This project **reuses the existing `churn-db` RDS MySQL instance** rather
than standing up a new database. You will not be creating any new RDS
resources — only a logical database named `tenk` inside that instance.

The MVP stack is small: one scorer **Lambda** (container image) behind an
**HTTP API Gateway**, plus a **Streamlit dashboard** running as a single
Fargate task. FinBERT runs on a pre-deployed **SageMaker endpoint**.
XGBoost is bundled inside the Lambda image — no second endpoint.

---

## 0. One-time account prep

1. **Sign in** to https://console.aws.amazon.com.
2. Top-right region switcher → pick the region `churn-db` lives in (likely
   **us-east-1**).
3. Open **CloudShell**: click the terminal icon in the top toolbar (right of
   the search bar).

---

## 1. Gather info about the existing RDS + VPC (**[Console]**)

1. **RDS → Databases → `churn-db`**. Copy:
   - **Endpoint** (e.g. `churn-db.xxxxx.us-east-1.rds.amazonaws.com`)
   - **VPC**: `vpc-038924d5598e6b875`
   - **VPC security group(s)**: the SG attached to the cluster (e.g. `sg-0xxxxx`).
     You'll add an inbound rule here in step 4.
   - **Master username**
   - **Master password** (from wherever you stored it)
2. **VPC → Subnets** → filter by `vpc-038924d5598e6b875` → note:
   - 2+ subnet IDs for **private** (Lambda will attach ENIs here)
   - 2+ subnet IDs for **public** (dashboard task, needs internet)
   - In a default VPC, all subnets are public — reuse the same list for both.

---

## 2. Clone the repo into CloudShell

```bash
git clone <your-repo-url> project && cd project
chmod +x scripts/*.sh scripts/*.py
```

---

## 3. Deploy the app stack (builds 2 container images + deploys CloudFormation)

`deploy.sh` builds and pushes two container images (scorer Lambda and
Streamlit dashboard), applies the MySQL schema, deploys the CloudFormation
stack, then rolls the ECS service.

> **Why is FinBERT baked into the Lambda?** Learner-Lab accounts deny
> `ec2:RunInstances` and `servicediscovery:CreatePrivateDnsNamespace`.
> The simplest path: bundle PyTorch CPU + FinBERT weights directly in the
> scorer image. No separate FinBERT service needed at all.

> **Note on build time:** the scorer image is ~1.5 GB (PyTorch CPU +
> FinBERT weights + XGBoost). The first build takes ~10 min in CloudShell.
> Subsequent deploys reuse the layer cache and are much faster.

```bash
scripts/deploy.sh \
    vpc-038924d5598e6b875 \
    subnet-AA,subnet-BB \
    subnet-CC,subnet-DD \
    churn-db.xxxxx.us-east-1.rds.amazonaws.com \
    <db-username> '<db-password>'
```

Expected tail output:

```
API endpoint:  https://xxxxx.execute-api.us-east-1.amazonaws.com
Lambda SG:     sg-0abc123...
IMPORTANT: on the existing RDS security group, add an inbound rule:
   TCP 3306  source=sg-0abc123...
```

**Save the values.** The API URL is what the dashboard talks to; the
Lambda SG ID is what you'll open up on RDS in the next step.

If the `mysql` step fails with "Can't connect to MySQL", see step 4
first — CloudShell's IP needs to be whitelisted. Then re-run with
`SKIP_SCHEMA=1 scripts/deploy.sh ...` (or without it, now that the IP is
allowed). CloudFormation is idempotent — safe to re-run.

---

## 4. Open the RDS security group to the Lambda **and** CloudShell

**[Console] → EC2 → Security Groups** → find the SG attached to `churn-db`
(from step 1) → **Inbound rules → Edit**. Add **two** rules:

| Type | Protocol | Port | Source | Purpose |
|---|---|---|---|---|
| MySQL/Aurora | TCP | 3306 | `<Lambda SG ID from step 3>` | Lets the scorer Lambda reach MySQL |
| MySQL/Aurora | TCP | 3306 | `<your CloudShell public IP>/32` | Lets the deploy script load the schema |

Find your CloudShell public IP with:

```bash
curl -s https://checkip.amazonaws.com
```

Once the schema is loaded you can delete the CloudShell IP rule; only the
Lambda SG rule is required long-term.

---

## 5. Confirm the stack is healthy (**[Console]**)

1. **CloudFormation → Stacks → tenk → Outputs** → `ApiEndpoint`,
   `LambdaSecurityGroupId`, `DashboardServiceName`.
2. **ECS → Clusters → tenk-cluster → Services → tenk-dashboard** → running 1/1.
3. **Lambda → Functions → tenk-scorer** → exists, image URI set, memory = 8192 MB.
4. **RDS Query Editor** (or `mysql` from CloudShell):
   ```sql
   USE tenk; SHOW TABLES;
   ```
   → should show one table: `scored_filings`.

---

## 6. Smoke-test the API

No external FinBERT service to wait for — the model runs inside the
Lambda. Run immediately after the stack finishes:

```bash
API=$(aws cloudformation describe-stacks --stack-name tenk \
    --query "Stacks[0].Outputs[?OutputKey=='ApiEndpoint'].OutputValue" \
    --output text)
curl "$API/tickers/AAPL/filings"
# Pick an accession from the response, then:
curl "$API/tickers/AAPL/analysis?accession=0000320193-24-000123"
```

First scoring of a fresh 10-K takes ~15–25 s (FinBERT runs in-process
inside the Lambda). Subsequent calls hit the MySQL cache and return
instantly. A Lambda cold start (first call after a long idle) adds ~20 s
for model load — normal for demo use.

---

## 7. Open the dashboard

1. **[Console] → ECS → Clusters → tenk-cluster → Tasks** → click the
   running task → copy its **Public IP**.
2. Open `http://<task-public-ip>:8501`.
3. Enter a ticker → select a 10-K from the dropdown → **Analyze**.

For a stable URL, put the service behind an Application Load Balancer (not
wired up in this template — can add later).

---

## Tearing it all down

```bash
aws cloudformation delete-stack --stack-name tenk
```

This removes the FinBERT Fargate service, the scorer Lambda, API Gateway,
ECS cluster, Cloud Map namespace, and all the SGs. The `churn-db` RDS
instance is untouched. To remove the logical database:

```sql
DROP DATABASE tenk;
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `deploy.sh` fails on `mysql` step | Add your CloudShell public IP to the RDS security group (step 4), then re-run. |
| Lambda logs show `Can't connect to MySQL` | Confirm the Lambda SG is on the RDS SG's inbound list (TCP 3306). Check CloudWatch `/aws/lambda/tenk-scorer`. |
| Lambda logs show `FinBERT error` or import failure | Check `/aws/lambda/tenk-scorer` in CloudWatch. Most likely the scorer image was built without the `transformers` layer — rebuild with the updated Dockerfile. |
| API returns 504 (timeout) | Filing is unusually large. Very long 10-Ks can push past the 29 s API GW limit even with aspect-tag filtering. Try a shorter 10-K first; cached results are instant. Lambda cold start (first call after idle) also adds ~20 s for model load — retry once. |
| API returns 500 | `/aws/lambda/tenk-scorer` CloudWatch logs. Usual culprits: DB creds wrong, FinBERT URL unreachable, SG not on RDS inbound. |
