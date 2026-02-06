# Agent Researcher - GCP Deployment Guide

## Infrastructure

- **GCP Project:** `prj-cts-lab-vertex-sandbox`
- **Region:** `us-east1`
- **Database:** Cloud SQL PostgreSQL (`dpe-db` at `34.74.169.146`)

## Cloud Run Services

### Backend

```bash
gcloud run deploy agent-researcher-backend \
  --project=prj-cts-lab-vertex-sandbox \
  --region=us-east1 \
  --image=us-east1-docker.pkg.dev/prj-cts-lab-vertex-sandbox/cloud-run-source-deploy/agent-researcher-backend:v5 \
  --port=8080 \
  --allow-unauthenticated \
  --memory=2Gi \
  --timeout=300 \
  --set-env-vars="DJANGO_SETTINGS_MODULE=backend.settings.prod,DATABASE_URL=postgres://postgres:<PASSWORD>@34.74.169.146:5432/agent_researcher,GEMINI_API_KEY=<GEMINI_KEY>,ALLOWED_HOSTS=*,CHROMA_PERSIST_DIR=/tmp/chromadb,CORS_ALLOWED_ORIGINS=https://agent-researcher-frontend-841327020312.us-east1.run.app,SECRET_KEY=<SECRET_KEY>"
```

**Key Settings:**
- **Memory: 2Gi** — Required for Google Search grounding feature
- **Timeout: 300s** — Research workflow can take 2-4 minutes
- **Port: 8080** — Django/Gunicorn default

### Frontend

```bash
gcloud run deploy agent-researcher-frontend \
  --project=prj-cts-lab-vertex-sandbox \
  --region=us-east1 \
  --image=us-east1-docker.pkg.dev/prj-cts-lab-vertex-sandbox/cloud-run-source-deploy/agent-researcher-frontend:v4 \
  --port=3000 \
  --allow-unauthenticated \
  --memory=256Mi
```

## Service URLs

- **Backend:** https://agent-researcher-backend-841327020312.us-east1.run.app
- **Frontend:** https://agent-researcher-frontend-841327020312.us-east1.run.app

## Database Migrations

After deploying a new backend version with schema changes:

```bash
# Update the migration job to use the new image
gcloud run jobs update agent-researcher-migrate \
  --project=prj-cts-lab-vertex-sandbox \
  --region=us-east1 \
  --image=us-east1-docker.pkg.dev/prj-cts-lab-vertex-sandbox/cloud-run-source-deploy/agent-researcher-backend:v5

# Run migrations
gcloud run jobs execute agent-researcher-migrate \
  --project=prj-cts-lab-vertex-sandbox \
  --region=us-east1 \
  --wait
```

## Build & Deploy Workflow

### 1. Build Docker Images

```bash
# Backend
cd backend
gcloud builds submit \
  --project=prj-cts-lab-vertex-sandbox \
  --region=us-east1 \
  --tag=us-east1-docker.pkg.dev/prj-cts-lab-vertex-sandbox/cloud-run-source-deploy/agent-researcher-backend:v5

# Frontend
cd frontend
gcloud builds submit \
  --project=prj-cts-lab-vertex-sandbox \
  --region=us-east1 \
  --tag=us-east1-docker.pkg.dev/prj-cts-lab-vertex-sandbox/cloud-run-source-deploy/agent-researcher-frontend:v4
```

### 2. Deploy Services

Run the deploy commands above.

### 3. Run Migrations

If there are new migrations, run the migration job.

## Fixing Stuck Research Jobs

If research jobs get stuck on "running" status:

```bash
gcloud run jobs execute fix-status \
  --project=prj-cts-lab-vertex-sandbox \
  --region=us-east1 \
  --wait
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `DJANGO_SETTINGS_MODULE` | `backend.settings.prod` for production |
| `DATABASE_URL` | PostgreSQL connection string |
| `GEMINI_API_KEY` | Google Gemini API key |
| `ALLOWED_HOSTS` | Django allowed hosts (`*` for Cloud Run) |
| `CHROMA_PERSIST_DIR` | ChromaDB storage path |
| `CORS_ALLOWED_ORIGINS` | Frontend URL for CORS |
| `SECRET_KEY` | Django secret key |

## Memory Requirements

| Feature | Min Memory |
|---------|------------|
| Basic research | 1Gi |
| Google Search grounding | 2Gi |
| PDF export | 2Gi |

**Current setting: 2Gi** to support all features.
