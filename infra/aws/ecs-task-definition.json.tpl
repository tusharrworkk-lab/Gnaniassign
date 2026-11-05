{
  "family": "gnani-phone-assistant",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "${ECS_EXECUTION_ROLE_ARN}",
  "taskRoleArn": "${ECS_TASK_ROLE_ARN}",
  "containerDefinitions": [
    {
      "name": "api",
      "image": "${IMAGE_URI}",
      "essential": true,
      "portMappings": [{"containerPort": 8080, "protocol": "tcp"}],
      "environment": [
        {"name": "AGENT_ENVIRONMENT", "value": "production"},
        {"name": "LLM_PROVIDER", "value": "bedrock_gpt_oss"},
        {"name": "BEDROCK_REGION", "value": "${AWS_REGION}"},
        {"name": "BEDROCK_GPT_OSS_MODEL", "value": "openai.gpt-oss-120b-1:0"},
        {"name": "INTENT_PROVIDER", "value": "generic"},
        {"name": "RETAIL_MINILM_ENABLED", "value": "false"},
        {"name": "PROMPT_TREE_ENABLED", "value": "true"},
        {"name": "PROMPT_CACHE_PROVIDER", "value": "none"},
        {"name": "TENANT_ID", "value": "default"},
        {"name": "AGENT_CHANNEL", "value": "voice"},
        {"name": "AGENT_LANGUAGE", "value": "hinglish"},
        {"name": "CONVERSATION_LOGGING_ENABLED", "value": "true"}
      ],
      "secrets": [
        {"name": "EPISODIC_MEMORY_DATABASE_URL", "valueFrom": "${DATABASE_URL_SECRET_ARN}"},
        {"name": "CONVERSATION_LOG_DATABASE_URL", "valueFrom": "${DATABASE_URL_SECRET_ARN}"},
        {"name": "KNOWLEDGE_BASE_DATABASE_URL", "valueFrom": "${DATABASE_URL_SECRET_ARN}"},
        {"name": "ORDER_DATA_DATABASE_URL", "valueFrom": "${DATABASE_URL_SECRET_ARN}"},
        {"name": "PROMPT_TREE_DATABASE_URL", "valueFrom": "${DATABASE_URL_SECRET_ARN}"},
        {"name": "GNANI_API_KEY", "valueFrom": "${GNANI_API_KEY_SECRET_ARN}"},
        {"name": "AGENT_API_KEY", "valueFrom": "${AGENT_API_KEY_SECRET_ARN}"},
        {"name": "WEBHOOK_SIGNING_SECRET", "valueFrom": "${WEBHOOK_SIGNING_SECRET_ARN}"}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "${CLOUDWATCH_LOG_GROUP}",
          "awslogs-region": "${AWS_REGION}",
          "awslogs-stream-prefix": "api"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "python -c \"import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=3)\""],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 30
      }
    }
  ]
}
