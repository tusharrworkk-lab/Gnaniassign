# Private Laya Lambda

This deploys the `multilingual` laya.cpp checkpoint as a private Lambda container in Mumbai (`ap-south-1`). The client uses `lambda:InvokeFunction` directly; it does not expose a Function URL or API Gateway.

## Important trade-off

Lambda has no GPU. laya.cpp supports CPU inference, but this is a low-traffic/cost-optimized deployment, not the lowest-latency option. The image embeds roughly 644 MB of model weights. Cold starts must load that model and can take seconds. A warm Lambda process keeps the native server and checkpoint resident.

Use `ReservedConcurrentExecutions: 2` as supplied to cap surprise spend. Do not enable provisioned concurrency until a measured warm-latency requirement justifies its continuous cost.

## Deploy

Install Docker, AWS CLI v2, and AWS SAM CLI. Authenticate to the intended AWS account and Mumbai region, then run from the repository root:

```bash
sam build --template-file infra/laya-lambda/template.yaml
sam deploy --guided --region ap-south-1 --capabilities CAPABILITY_IAM
```

SAM will build the Linux x86_64 CPU image, create an ECR repository, upload the image, and create the Lambda function. The first build is large because it contains the multilingual checkpoint.

The identity running your voice application needs this minimum permission, scoped to the deployed function ARN:

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": "lambda:InvokeFunction",
    "Resource": "arn:aws:lambda:ap-south-1:ACCOUNT_ID:function:gnani-laya-multilingual"
  }]
}
```

After deployment, set these values in the application's ignored `.env`:

```dotenv
INTENT_PROVIDER=laya_lambda
LAYA_LAMBDA_FUNCTION=gnani-laya-multilingual
LAYA_LAMBDA_REGION=ap-south-1
LAYA_LAMBDA_TIMEOUT_SECONDS=15
```

Benchmark at least 30 warm invokes and 10 forced cold invokes. Compare p50/p95 and cost per decision with your current local GPU Laya baseline before routing production voice turns through Lambda.
