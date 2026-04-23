#!/usr/bin/env bash
# End-to-end deploy script for the MVP stack.
#
# What it does, in order:
#   1. Ensure ECR repos exist
#   2. Build + push the scorer Lambda container image
#      (includes bundled XGBoost model AND FinBERT weights)
#   3. Build + push the Streamlit dashboard container image
#   4. Apply the MySQL schema against the existing churn-db instance
#   5. Deploy the CloudFormation stack (Lambda, API Gateway, ECS cluster+service)
#   6. Force-redeploy Lambda + ECS dashboard so they pick up the new images
#
# Prereqs:
#   - AWS CLI configured (or running in CloudShell)
#   - Docker running
#   - A pre-existing IAM role named LabRole (AWS Academy / Learner Lab). All
#     services in this stack (Lambda, ECS task, ECS task-execution) assume
#     that single role — we do not create any IAM resources. Override with
#     LAB_ROLE_NAME or LAB_ROLE_ARN if your role has a different name.
#   - RDS instance reachable from this machine (to apply the schema). If not,
#     apply sql/schema.sql via the RDS query editor / Session Manager instead,
#     and pass SKIP_SCHEMA=1.
#   - After first deploy, add an inbound rule on the RDS SG allowing TCP 3306
#     from the LambdaSecurityGroup that this stack outputs.
#
# Usage:
#   scripts/deploy.sh <vpc-id> <public-subnet-ids-csv> \
#                     <db-host> <db-username> <db-password> [db-name]

set -euo pipefail

PROJECT="${PROJECT:-tenk}"
REGION="${AWS_REGION:-us-east-1}"
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)

VPC_ID="${1:?vpc-id}"
PUBLIC_SUBNETS="${2:?public subnets csv}"
DB_HOST="${3:?db host (RDS endpoint)}"
DB_USER="${4:?db username}"
DB_PASSWORD="${5:?db password}"
DB_NAME="${6:-tenk}"

HERE="$(cd "$(dirname "$0")/.." && pwd)"
ECR_HOST="$ACCOUNT.dkr.ecr.$REGION.amazonaws.com"
SCORER_REPO="$PROJECT-scorer"
DASHBOARD_REPO="$PROJECT-dashboard"
IMAGE_TAG="${IMAGE_TAG:-$(date +%Y%m%d%H%M%S)}"

# Reuse the pre-existing LabRole for every service in the stack.
LAB_ROLE_NAME="${LAB_ROLE_NAME:-LabRole}"
LAB_ROLE_ARN="${LAB_ROLE_ARN:-$(aws iam get-role --role-name "$LAB_ROLE_NAME" \
    --query 'Role.Arn' --output text)}"
if [ -z "$LAB_ROLE_ARN" ] || [ "$LAB_ROLE_ARN" = "None" ]; then
    echo "Could not resolve IAM role '$LAB_ROLE_NAME'. Set LAB_ROLE_ARN explicitly." >&2
    exit 1
fi
echo "==> Using IAM role: $LAB_ROLE_ARN"

echo "==> ECR login"
for repo in "$SCORER_REPO" "$DASHBOARD_REPO"; do
    aws ecr describe-repositories --repository-names "$repo" --region "$REGION" >/dev/null 2>&1 \
        || aws ecr create-repository --repository-name "$repo" --region "$REGION" >/dev/null
done
aws ecr get-login-password --region "$REGION" \
    | docker login --username AWS --password-stdin "$ECR_HOST"

echo "==> Build + push scorer image ($IMAGE_TAG)  [large: ~1.5 GB, may take ~10 min first time]"
echo "    (includes PyTorch CPU + FinBERT weights + XGBoost model)"
docker build --platform linux/amd64 --provenance=false \
    -t "$SCORER_REPO:$IMAGE_TAG" \
    -f "$HERE/lambdas/scorer/Dockerfile" "$HERE"
docker tag "$SCORER_REPO:$IMAGE_TAG" "$ECR_HOST/$SCORER_REPO:$IMAGE_TAG"
docker push "$ECR_HOST/$SCORER_REPO:$IMAGE_TAG"
SCORER_URI="$ECR_HOST/$SCORER_REPO:$IMAGE_TAG"

echo "==> Build + push dashboard image ($IMAGE_TAG)"
docker build --platform linux/amd64 --provenance=false \
    -t "$DASHBOARD_REPO:$IMAGE_TAG" \
    -f "$HERE/dashboard/Dockerfile" "$HERE"
docker tag "$DASHBOARD_REPO:$IMAGE_TAG" "$ECR_HOST/$DASHBOARD_REPO:$IMAGE_TAG"
docker push "$ECR_HOST/$DASHBOARD_REPO:$IMAGE_TAG"
DASHBOARD_URI="$ECR_HOST/$DASHBOARD_REPO:$IMAGE_TAG"

if [ "${SKIP_SCHEMA:-0}" != "1" ]; then
    echo "==> Apply MySQL schema to $DB_HOST/$DB_NAME"
    mysql -h "$DB_HOST" -u "$DB_USER" -p"$DB_PASSWORD" \
        -e "CREATE DATABASE IF NOT EXISTS \`$DB_NAME\` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
    mysql -h "$DB_HOST" -u "$DB_USER" -p"$DB_PASSWORD" "$DB_NAME" < "$HERE/sql/schema.sql"
else
    echo "==> SKIP_SCHEMA=1 → skipping schema apply (do it manually)"
fi

echo "==> Deploy CloudFormation stack"
aws cloudformation deploy \
    --template-file "$HERE/infrastructure/cloudformation.yaml" \
    --stack-name "$PROJECT" \
    --region "$REGION" \
    --parameter-overrides \
        ProjectName="$PROJECT" \
        VpcId="$VPC_ID" \
        PublicSubnetIds="$PUBLIC_SUBNETS" \
        ScorerImageUri="$SCORER_URI" \
        DashboardImageUri="$DASHBOARD_URI" \
        LabRoleArn="$LAB_ROLE_ARN" \
        DBHost="$DB_HOST" \
        DBName="$DB_NAME" \
        DBUsername="$DB_USER" \
        DBPassword="$DB_PASSWORD"

echo "==> Force new deployment (Lambda + ECS dashboard) to pick up fresh images"
aws lambda update-function-code \
    --function-name "$PROJECT-scorer" \
    --image-uri "$SCORER_URI" \
    --region "$REGION" >/dev/null
aws ecs update-service \
    --cluster "$PROJECT-cluster" --service "$PROJECT-dashboard" \
    --force-new-deployment --region "$REGION" >/dev/null

API_URL=$(aws cloudformation describe-stacks --stack-name "$PROJECT" \
    --query "Stacks[0].Outputs[?OutputKey=='ApiEndpoint'].OutputValue" --output text --region "$REGION")

cat <<EOF

==> Done.

API endpoint:  $API_URL

IMPORTANT: Lambda now runs outside the VPC to reach EDGAR on the internet.
Open the RDS security group inbound rule TCP 3306 to 0.0.0.0/0
(or restrict to Lambda's IP range) so it can reach MySQL.

Smoke test:
   curl "$API_URL/tickers/AAPL/filings"
   curl "$API_URL/tickers/AAPL/analysis?accession=<pick-from-above>"

First scoring of a fresh filing takes ~15-25 s (FinBERT runs in-process).
Subsequent calls return instantly from the MySQL cache.

EOF
