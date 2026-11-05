#!/usr/bin/env bash
# Deploy one immutable container image on the existing GCP VM. This script is
# copied and invoked by GitHub Actions over IAP; secrets remain in the VM's
# ignored .env.local-cloud file and never enter Actions logs.
set -Eeuo pipefail

image_uri=${1:?Usage: deploy-gcp-vm.sh <artifact-registry-image-uri>}
app_dir=${VOICE_AGENT_APP_DIR:-/opt/voice-agent/app}
compose=(sudo docker compose --env-file .env.local-cloud -f docker-compose.local-cloud.yml -f docker-compose.production.yml)

cd "$app_dir"
previous_image=$(sudo docker inspect --format '{{.Config.Image}}' gnani-local-cloud-api-1 2>/dev/null || true)
registry_host=${image_uri%%/*}

# The VM's attached service account is granted Artifact Registry Reader during
# bootstrap. Docker uses gcloud's credential helper; no registry password is
# written into the repository or GitHub.
gcloud auth configure-docker "$registry_host" --quiet

rollback() {
  exit_code=$?
  if [[ -n "$previous_image" ]]; then
    echo "Deployment failed; restoring the prior API image."
    AGENT_IMAGE="$previous_image" "${compose[@]}" up -d --no-build --force-recreate api || true
  fi
  exit "$exit_code"
}
trap rollback ERR

AGENT_IMAGE="$image_uri" "${compose[@]}" pull api
AGENT_IMAGE="$image_uri" "${compose[@]}" up -d --no-build --force-recreate api

for attempt in $(seq 1 18); do
  if curl --fail --silent --show-error http://127.0.0.1:8080/healthz >/dev/null; then
    trap - ERR
    echo "Deployment healthy: $image_uri"
    exit 0
  fi
  sleep 5
done

echo "The new API image did not become healthy."
false
