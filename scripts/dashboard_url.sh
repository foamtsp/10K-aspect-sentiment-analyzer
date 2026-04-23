#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-tenk}"
REGION="${AWS_REGION:-us-east-1}"

TASK=$(aws ecs list-tasks \
    --cluster "$PROJECT-cluster" \
    --service-name "$PROJECT-dashboard" \
    --region "$REGION" \
    --query 'taskArns[0]' --output text)

if [ -z "$TASK" ] || [ "$TASK" = "None" ]; then
    echo "No running dashboard task found in $PROJECT-cluster." >&2
    exit 1
fi

ENI=$(aws ecs describe-tasks \
    --cluster "$PROJECT-cluster" \
    --tasks "$TASK" \
    --region "$REGION" \
    --query 'tasks[0].attachments[0].details[?name==`networkInterfaceId`].value' \
    --output text)

PUBLIC_IP=$(aws ec2 describe-network-interfaces \
    --network-interface-ids "$ENI" \
    --region "$REGION" \
    --query 'NetworkInterfaces[0].Association.PublicIp' \
    --output text)

echo "http://$PUBLIC_IP:8501"
