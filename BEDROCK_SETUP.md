# Amazon Bedrock setup

This guide configures a separate Python project to call a foundation model through Amazon Bedrock. It uses the Bedrock Runtime `ConverseStream` API so the application can receive text incrementally.

The examples use the Mumbai region (`ap-south-1`). Change the region and model ID if the target project uses another region.

## 1. Prerequisites

- An AWS account with Bedrock enabled in the target region.
- Access to the model you want to use in that region.
- Python 3.10 or newer.
- Either a Bedrock API key or AWS IAM credentials.

Confirm that the model ID is available in the selected region before putting it in configuration. The example below uses:

```text
google.gemma-3-4b-it
```

Model availability, quotas, and access requirements can vary by account and region.

## 2. Install the SDK

For a normal Python project:

```bash
python -m pip install boto3 python-dotenv
```

For `uv`:

```bash
uv add boto3 python-dotenv
```

## 3. Configure environment variables

Create a local `.env` file and do not commit it:

```dotenv
BEDROCK_REGION=ap-south-1
BEDROCK_MODEL=google.gemma-3-4b-it
BEDROCK_MAX_OUTPUT_TOKENS=256
BEDROCK_TEMPERATURE=0.2
```

Add `.env` to `.gitignore`:

```gitignore
.env
```

### Option A: Bedrock API key

Set the Bedrock API key in the environment used by the process:

```dotenv
BEDROCK_API_KEY=replace_with_your_key
```

The AWS SDK reads this credential as a Bedrock bearer token when it is exposed as `AWS_BEARER_TOKEN_BEDROCK`. The sample application below maps `BEDROCK_API_KEY` to that variable without writing the key to disk.

### Option B: IAM or AWS SSO

Do not set `BEDROCK_API_KEY`. Configure the standard AWS credential chain instead.

For AWS SSO:

```bash
aws configure sso --profile bedrock-mumbai
aws sso login --profile bedrock-mumbai
```

Then add the profile to `.env`:

```dotenv
AWS_PROFILE=bedrock-mumbai
```

In deployed environments, prefer an IAM role attached to the workload instead of long-lived access keys.

## 4. Required IAM permissions

Grant the runtime identity only the model actions it needs. Replace the model ARN account and region values as appropriate:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream"
      ],
      "Resource": "arn:aws:bedrock:ap-south-1::foundation-model/google.gemma-3-4b-it"
    }
  ]
}
```

Some models or account configurations require additional model-access approval in the Bedrock console. An IAM permission alone does not guarantee that the model can be invoked.

## 5. Minimal streaming client

Create `bedrock_client.py`:

```python
import os

import boto3
from dotenv import load_dotenv


load_dotenv()

REGION = os.getenv("BEDROCK_REGION", "ap-south-1")
MODEL = os.getenv("BEDROCK_MODEL", "google.gemma-3-4b-it")


def create_client():
    # Use the Bedrock API-key path only when a key is explicitly configured.
    api_key = os.getenv("BEDROCK_API_KEY")
    if api_key:
        os.environ["AWS_BEARER_TOKEN_BEDROCK"] = api_key

    profile = os.getenv("AWS_PROFILE")
    session = boto3.Session(
        profile_name=profile or None,
        region_name=REGION,
    )
    return session.client("bedrock-runtime")


def stream_answer(user_text: str, system_text: str = "You are a concise helpful assistant."):
    client = create_client()
    response = client.converse_stream(
        modelId=MODEL,
        system=[{"text": system_text}],
        messages=[
            {
                "role": "user",
                "content": [{"text": user_text}],
            }
        ],
        inferenceConfig={
            "maxTokens": int(os.getenv("BEDROCK_MAX_OUTPUT_TOKENS", "256")),
            "temperature": float(os.getenv("BEDROCK_TEMPERATURE", "0.2")),
        },
    )

    for event in response["stream"]:
        delta = event.get("contentBlockDelta", {}).get("delta", {})
        text = delta.get("text")
        if text:
            yield text


if __name__ == "__main__":
    for chunk in stream_answer("What can you help me with?"):
        print(chunk, end="", flush=True)
    print()
```

Run it with:

```bash
python bedrock_client.py
```

## 6. Non-streaming request

If streaming is not needed, use `client.converse()` instead:

```python
response = create_client().converse(
    modelId=MODEL,
    messages=[
        {
            "role": "user",
            "content": [{"text": "What can you help me with?"}],
        }
    ],
)

answer = response["output"]["message"]["content"][0]["text"]
print(answer)
```

## 7. Verify the configuration

Check the AWS identity first:

```bash
aws sts get-caller-identity --profile bedrock-mumbai
```

Then run the Python smoke test:

```bash
python bedrock_client.py
```

For an IAM setup, also verify that the selected region is correct:

```bash
AWS_PROFILE=bedrock-mumbai aws configure get region
```

## Troubleshooting

### `AccessDeniedException`

- Confirm the IAM policy includes both `bedrock:InvokeModel` and `bedrock:InvokeModelWithResponseStream`.
- Confirm the policy resource matches the selected model and region.
- Confirm model access is enabled for the account.

### `ResourceNotFoundException` or model not found

- Check the exact model ID.
- Check that the model is available in `BEDROCK_REGION`.
- Do not substitute a global or inference-profile ID when the application expects a foundation-model ID.

### Credentials are not found

- For SSO, run `aws sso login --profile <profile>`.
- Confirm `AWS_PROFILE` is visible to the Python process.
- In AWS compute, prefer the attached task, instance, or Lambda role.
- For API-key mode, confirm `BEDROCK_API_KEY` is set and that the process maps it to `AWS_BEARER_TOKEN_BEDROCK`.

### The request succeeds but output is empty

Inspect the complete response event shape for the selected model. Most Converse-compatible models stream text under `contentBlockDelta.delta.text`, but model-specific metadata events may also be present.

## Production recommendations

- Keep credentials in a secret manager, workload identity, or IAM role.
- Set request timeouts and retry limits in the client configuration.
- Cap `maxTokens` for latency-sensitive voice or chat responses.
- Log latency, model ID, and request IDs, but never log prompts containing sensitive data or credentials.
- Monitor Bedrock quotas and cost by model and region.
