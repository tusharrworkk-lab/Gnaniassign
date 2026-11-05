# GCP VM CI/CD: GitHub Actions → Artifact Registry → VM

This is the production path for `origin/master`:

```text
push to master → regression tests → immutable Docker image → Artifact Registry
               → IAP SSH deployment to voice-agent-01 → localhost health check
               → public HTTPS health check
```

The VM keeps `.env.local-cloud` locally. GitHub Actions never receives the
application, database, Bedrock, Gnani, or webhook secrets.

## One-time bootstrap

Run the commands below as a project owner after setting the values. They create
the Artifact Registry repository, a least-privilege GitHub deployment identity,
and a GitHub OIDC trust relationship. The workflow itself uses no long-lived
GCP key.

```bash
export GCP_PROJECT_ID=gen-lang-client-0813070445
export GCP_REGION=asia-south1
export GITHUB_ORG=tusharrworkk-lab
export GITHUB_REPO=Gnaniassign
export DEPLOY_SA=github-voice-agent-deployer
export WIF_POOL=github-actions
export WIF_PROVIDER=gnaniassign

gcloud artifacts repositories create voice-agent \
  --project="$GCP_PROJECT_ID" --location="$GCP_REGION" \
  --repository-format=docker --description="Voice agent production images"

gcloud iam service-accounts create "$DEPLOY_SA" --project="$GCP_PROJECT_ID" \
  --display-name="GitHub deployer for voice agent"

export DEPLOY_SA_EMAIL="$DEPLOY_SA@$GCP_PROJECT_ID.iam.gserviceaccount.com"
export PROJECT_NUMBER=$(gcloud projects describe "$GCP_PROJECT_ID" --format='value(projectNumber)')

gcloud iam workload-identity-pools create "$WIF_POOL" \
  --project="$GCP_PROJECT_ID" --location=global --display-name="GitHub Actions"
gcloud iam workload-identity-pools providers create-oidc "$WIF_PROVIDER" \
  --project="$GCP_PROJECT_ID" --location=global --workload-identity-pool="$WIF_POOL" \
  --display-name="Gnaniassign GitHub OIDC" \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.ref=assertion.ref" \
  --attribute-condition="assertion.repository=='$GITHUB_ORG/$GITHUB_REPO' && assertion.ref=='refs/heads/master'"

export WIF_PROVIDER_RESOURCE="projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/$WIF_POOL/providers/$WIF_PROVIDER"
gcloud iam service-accounts add-iam-policy-binding "$DEPLOY_SA_EMAIL" \
  --project="$GCP_PROJECT_ID" --role=roles/iam.workloadIdentityUser \
  --member="principalSet://iam.googleapis.com/projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/$WIF_POOL/attribute.repository/$GITHUB_ORG/$GITHUB_REPO"

gcloud projects add-iam-policy-binding "$GCP_PROJECT_ID" --member="serviceAccount:$DEPLOY_SA_EMAIL" --role=roles/artifactregistry.writer
gcloud projects add-iam-policy-binding "$GCP_PROJECT_ID" --member="serviceAccount:$DEPLOY_SA_EMAIL" --role=roles/compute.osAdminLogin
gcloud projects add-iam-policy-binding "$GCP_PROJECT_ID" --member="serviceAccount:$DEPLOY_SA_EMAIL" --role=roles/iap.tunnelResourceAccessor
```

Enable OS Login for the deployment service account and allow IAP to reach SSH
on the VM. Assign Artifact Registry Reader to the VM's own service account so
Docker can pull an image after deployment. The exact VM service-account email
is discovered with `gcloud compute instances describe`; do not guess it.

```bash
export VM_NAME=voice-agent-01
export GCP_ZONE=asia-south1-a
export VM_SA=$(gcloud compute instances describe "$VM_NAME" --project="$GCP_PROJECT_ID" --zone="$GCP_ZONE" --format='value(serviceAccounts[0].email)')

gcloud projects add-iam-policy-binding "$GCP_PROJECT_ID" --member="serviceAccount:$VM_SA" --role=roles/artifactregistry.reader
gcloud compute firewall-rules create voice-agent-allow-iap-ssh \
  --project="$GCP_PROJECT_ID" --network=default --direction=INGRESS --priority=800 \
  --action=ALLOW --rules=tcp:22 --source-ranges=35.235.240.0/20 --target-tags=voice-agent
```

## GitHub production environment

Create a protected **production** environment in the `Gnaniassign` repository.
Set these Environment variables:

```text
GCP_PROJECT_ID=gen-lang-client-0813070445
GCP_REGION=asia-south1
GCP_ARTIFACT_REPOSITORY=voice-agent
GCP_ZONE=asia-south1-a
GCP_VM_NAME=voice-agent-01
HEALTHCHECK_URL=https://voice.sidecut.me
```

Set these Environment secrets:

```text
GCP_WORKLOAD_IDENTITY_PROVIDER=projects/PROJECT_NUMBER/locations/global/workloadIdentityPools/github-actions/providers/gnaniassign
GCP_DEPLOY_SERVICE_ACCOUNT=github-voice-agent-deployer@gen-lang-client-0813070445.iam.gserviceaccount.com
```

After this setup, pushing to `origin/master` runs the workflow. The existing
AWS ECS workflow is not used by this GCP deployment and should be disabled in
the GitHub Actions UI to avoid a duplicate deployment.
