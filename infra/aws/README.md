# AWS deployment: GitHub Actions → ECR → ECS Fargate

This repository deploys the FastAPI webhook service to ECS Fargate. The
workflow does not create your VPC, RDS, Redis, ALB, domain, or IAM roles; those
are account infrastructure and should be created once in AWS.

## One-time AWS setup

1. Use `ap-south-1` unless you intentionally run Bedrock elsewhere.
2. Create an ECR repository, ECS cluster, Fargate service, ALB target group,
   HTTPS listener, CloudWatch log group, and Route 53/ACM hostname.
3. Configure the ALB target group as `HTTP`, port `8080`, target type `ip`,
   and health path `/healthz`.
4. Put ECS tasks in private subnets. Allow ALB → task security group on 8080;
   allow public internet → ALB only on 443.
5. Give the ECS **task role** permission to invoke the configured Bedrock model.
   Give the **execution role** permission to pull ECR images, write logs, and
   read the referenced Secrets Manager secrets.
6. Apply PostgreSQL schemas, including `prompt_trees.sql` and the active
   personal prompt-tree migration, to the RDS database.

## GitHub production environment

Create a protected `production` GitHub Environment. Add this secret:

```text
AWS_DEPLOY_ROLE_ARN
```

Configure GitHub Actions OIDC trust on that role; do not use permanent AWS
access keys. Add these GitHub Environment variables:

```text
AWS_REGION=ap-south-1
ECR_REPOSITORY=gnani-phone-assistant
ECS_CLUSTER=voice-agent
ECS_SERVICE=phone-assistant-api
HEALTHCHECK_URL=https://voice.example.com
ECS_EXECUTION_ROLE_ARN=arn:aws:iam::ACCOUNT:role/ecsTaskExecutionRole
ECS_TASK_ROLE_ARN=arn:aws:iam::ACCOUNT:role/phone-assistant-task-role
DATABASE_URL_SECRET_ARN=arn:aws:secretsmanager:...
GNANI_API_KEY_SECRET_ARN=arn:aws:secretsmanager:...
AGENT_API_KEY_SECRET_ARN=arn:aws:secretsmanager:...
WEBHOOK_SIGNING_SECRET_ARN=arn:aws:secretsmanager:...
CLOUDWATCH_LOG_GROUP=/ecs/gnani-phone-assistant
```

The service must exist before the first deployment so the workflow can capture
its previous task definition for rollback. The first task definition can be
rendered from `ecs-task-definition.json.tpl` locally or through the GitHub
workflow after creating an initial service revision.

After deployment, give your telephony provider:

```text
https://voice.example.com/v1/webhooks/call
```
