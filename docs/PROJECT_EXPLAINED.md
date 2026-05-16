# Complete Project Explanation: Deploying a Machine Learning Model with GitHub Actions and Hugging Face Spaces

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [Overall Architecture](#2-overall-architecture)
3. [Folder Structure Explained](#3-folder-structure-explained)
4. [The Machine Learning Model](#4-the-machine-learning-model)
5. [app/Dockerfile — Building the Docker Image](#5-appdockerfile--building-the-docker-image)
6. [app/requirements.txt — Python Dependencies](#6-apprequirementstxt--python-dependencies)
7. [app/main.py — The API Server](#7-appmainpy--the-api-server)
8. [hf-space/Dockerfile — The Deployment Pointer](#8-hf-spacedockerfile--the-deployment-pointer)
9. [tests/test_app.py — Reference File](#9-teststest_apppy--reference-file)
10. [.github/workflows/deploy.yml — The CI/CD Pipeline](#10-githubworkflowsdeployyml--the-cicd-pipeline)
11. [Hugging Face Spaces — Where the Model Runs](#11-hugging-face-spaces--where-the-model-runs)
12. [Docker Hub — Where the Image Lives](#12-docker-hub--where-the-image-lives)
13. [GPU or CPU? What Hardware Is Used?](#13-gpu-or-cpu-what-hardware-is-used)
14. [Model Versioning — v1 vs v2](#14-model-versioning--v1-vs-v2)
15. [GitHub Secrets — How Credentials Are Managed](#15-github-secrets--how-credentials-are-managed)
16. [End-to-End Flow: What Happens When You Push Code](#16-end-to-end-flow-what-happens-when-you-push-code)
17. [Key Decisions and Why](#17-key-decisions-and-why)

---

## 1. What This Project Does

This project demonstrates a complete **MLOps CI/CD pipeline** using GitHub Actions to automatically deploy a machine learning model to Hugging Face Spaces.

**In simple terms:**
- A sentiment analysis model (DistilBERT) is packaged inside a Docker image
- That Docker image is stored on Docker Hub — built once manually, never by the pipeline
- Every time a developer pushes code to GitHub, the pipeline automatically:
  - Pulls the real Docker image and runs a live prediction to confirm the model works
  - Deploys the model to Hugging Face Spaces
  - Confirms the live endpoint is healthy

**The key MLOps principle demonstrated:** No one manually logs into a server, copies files, or restarts services. One `git push` triggers everything automatically.

---

## 2. Overall Architecture

```
Developer pushes code
        │
        ▼
   GitHub Repository
        │
        ▼
  GitHub Actions (CI/CD)
        │
        ▼
  smoke-test ──────────── Docker Hub
  (pull image, run it,    (image pulled from here)
   test real prediction)
        │ passes
        ▼
      deploy ──────────── Hugging Face Spaces
        │                 (model deployed here)
        ▼
   health-check ──────── Live API endpoint confirmed
```

**Three external services are involved:**

| Service | Role |
|---|---|
| **GitHub** | Stores code and runs the pipeline |
| **Docker Hub** | Stores the pre-built Docker image |
| **Hugging Face Spaces** | Runs the model as a live API |

---

## 3. Folder Structure Explained

```
github-action-model/
│
├── .github/
│   └── workflows/
│       └── deploy.yml        ← The CI/CD pipeline (3 jobs)
│
├── app/                      ← Used ONCE manually to build the Docker image
│   ├── main.py               ← FastAPI server with DistilBERT
│   ├── requirements.txt      ← Python dependencies
│   ├── Dockerfile            ← Instructions to build the image
│   └── __init__.py           ← Makes app/ a Python package
│
├── hf-space/
│   └── Dockerfile            ← 2-line file: points to Docker Hub image tag
│
├── tests/
│   └── test_app.py           ← Reference file (not used by pipeline)
│
├── docs/
│   └── PROJECT_EXPLAINED.md  ← This file
│
└── pytest.ini                ← Test configuration
```

**Critical distinction:**

| Folder | Purpose | Who uses it |
|---|---|---|
| `app/` | Build the Docker image | Developer, manually, once |
| `hf-space/` | Tell HF Spaces which image to run | Pipeline, on every push |
| `tests/` | Reference — not part of the active pipeline | For learning/reference |

---

## 4. The Machine Learning Model

**Model name:** `distilbert-base-uncased-finetuned-sst-2-english`

**What is DistilBERT?**
DistilBERT is a smaller, faster version of BERT (Bidirectional Encoder Representations from Transformers), a language model created by Google. "Distilled" means it was trained to mimic a larger model (BERT) while being 40% smaller and 60% faster, with only a 3% drop in accuracy.

**What task does it perform?**
Sentiment Analysis — given a sentence of text, it predicts whether the sentiment is POSITIVE or NEGATIVE, along with a confidence score (0.0 to 1.0).

**Example:**
```
Input:  "The weather today is absolutely wonderful!"
Output: {"label": "POSITIVE", "score": 0.9998, "model_version": "v2"}
```

**Where does the model come from?**
The model is NOT baked inside the Docker image. It lives on Hugging Face Hub at:
`https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english`

When the container starts, the `transformers` library automatically downloads the model weights (~250MB) from Hugging Face Hub into a local cache. This happens on first startup — which is why the pipeline waits for the container to be ready before testing it.

**Who trained this model?**
Hugging Face. It was fine-tuned on the SST-2 (Stanford Sentiment Treebank) dataset — a collection of movie reviews labeled as positive or negative.

---

## 5. app/Dockerfile — Building the Docker Image

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
```

**Line by line explanation:**

| Line | What it does |
|---|---|
| `FROM python:3.11-slim` | Starts from an official Python 3.11 base image. `slim` excludes documentation and unnecessary packages to keep the image smaller |
| `WORKDIR /app` | Sets the working directory inside the container. All subsequent commands run from here |
| `COPY requirements.txt .` | Copies `requirements.txt` from your machine into the container |
| `RUN pip install --no-cache-dir -r requirements.txt` | Installs all Python packages. `--no-cache-dir` avoids storing download cache inside the image |
| `COPY main.py .` | Copies the FastAPI application code into the container |
| `EXPOSE 7860` | Tells Docker and Hugging Face Spaces this container listens on port 7860 — HF Spaces requires this exact port |
| `CMD [...]` | Command that runs when the container starts. Launches the uvicorn web server on port 7860 |

**Why is this built manually and not by the pipeline?**
This image is ~8.67GB (PyTorch alone is ~779MB, plus CUDA libraries). Building it on every push would make the pipeline take 15+ minutes. Instead it is built once, pushed to Docker Hub permanently, and the pipeline just references it by tag.

**How to build and push manually:**
```bash
docker build -t 03sarath/distilbert-sentiment:v1 ./app
docker push 03sarath/distilbert-sentiment:v1
```

---

## 6. app/requirements.txt — Python Dependencies

```
fastapi==0.111.0
uvicorn==0.29.0
transformers==4.41.0
torch==2.3.0
pydantic==2.7.0
```

| Package | Purpose |
|---|---|
| `fastapi` | Web framework for building the REST API |
| `uvicorn` | ASGI web server that runs the FastAPI application |
| `transformers` | Hugging Face library providing DistilBERT and the `pipeline()` function |
| `torch` | PyTorch — the deep learning framework DistilBERT runs on |
| `pydantic` | Data validation used by FastAPI to validate request and response shapes |

**Why are exact versions pinned?**
Pinning (e.g., `torch==2.3.0` not just `torch`) ensures the Docker image always installs identical packages. Without pinning, a rebuild six months later might install a newer torch version that breaks the model.

---

## 7. app/main.py — The API Server

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

MODEL_VERSION = "v2"

app = FastAPI(title="DistilBERT Sentiment API")

classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    label: str
    score: float
    model_version: str

@app.get("/health")
def health():
    return {"status": "ok", "version": MODEL_VERSION}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="text cannot be empty")
    result = classifier(request.text)[0]
    return PredictResponse(
        label=result["label"],
        score=round(result["score"], 4),
        model_version=MODEL_VERSION
    )
```

**Section by section:**

**`MODEL_VERSION = "v2"`**
Identifies which version is running. Returned in every API response — this is how you confirm which version is deployed without looking at the server.

**`classifier = pipeline(...)`**
Runs at container startup. Downloads DistilBERT from Hugging Face Hub and loads it into memory. This is why the container takes 1–2 minutes to be ready after starting — it is downloading ~250MB of model weights.

**`class PredictRequest / PredictResponse`**
Define the exact shape of the API input and output. FastAPI uses these to automatically validate requests and generate documentation at `/docs`.

**`/health` endpoint**
Used by the pipeline's health-check job to confirm the service is alive. Returns `version` so it is immediately clear which version is deployed.

**`/predict` endpoint**
The main endpoint. The empty text check (`if not request.text.strip()`) rejects blank input before it reaches the model — this is important because sending empty text to DistilBERT causes an error.

---

## 8. hf-space/Dockerfile — The Deployment Pointer

```dockerfile
FROM 03sarath/distilbert-sentiment:v1

EXPOSE 7860
```

This 2-line file is the most important file for deployment — and it is intentionally this minimal.

**What it does:**
- `FROM 03sarath/distilbert-sentiment:v1` — tells Hugging Face Spaces which image to pull from Docker Hub and run
- `EXPOSE 7860` — tells Hugging Face the container uses port 7860

**Why is this separate from `app/Dockerfile`?**

| | `app/Dockerfile` | `hf-space/Dockerfile` |
|---|---|---|
| **Purpose** | Build the full image | Point to the built image |
| **Result** | ~8.67GB Docker image | 2-line deployment instruction |
| **When used** | Once, manually | Every push, by the pipeline |
| **Contains** | Python, packages, code, model loader | Nothing — just a reference |

**How version upgrades work — change one word and push:**
```dockerfile
# Change this
FROM 03sarath/distilbert-sentiment:v1
# To this
FROM 03sarath/distilbert-sentiment:v2
```
`git commit` + `git push origin main` → pipeline runs → new version deployed automatically.

**How rollback works:**
Change back to `v1` and push. The pipeline deploys v1 again in minutes.

---

## 9. tests/test_app.py — Reference File

```python
import sys
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

# torch and transformers are 3GB — replace with fakes so CI runs in seconds
mock_transformers = MagicMock()
mock_transformers.pipeline.return_value = MagicMock(
    return_value=[{"label": "POSITIVE", "score": 0.9998}]
)
sys.modules["transformers"] = mock_transformers
sys.modules["torch"] = MagicMock()

from app.main import app

client = TestClient(app)

# Gate 1: is the service reachable?
def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200

# Gate 2: does predict return the expected fields?
def test_predict_returns_sentiment():
    response = client.post("/predict", json={"text": "I love this course!"})
    assert response.status_code == 200
    assert "label" in response.json()
    assert "score" in response.json()

# Gate 3: is bad input rejected before reaching the model?
def test_empty_text_is_rejected():
    response = client.post("/predict", json={"text": "   "})
    assert response.status_code == 400
```

**Important: this file is NOT used by the active pipeline.**

The pipeline replaced the mocked test approach with a real model smoke test (Job 1 — smoke-test). This file is kept in the repo as a **reference** to show the alternative approach and for local development use.

**What it demonstrates conceptually:**
- `sys.modules["transformers"] = MagicMock()` — replaces the real 3GB library with a fake object so tests run in seconds without installing torch
- Tests check API routing, response shape, and input validation — not the model itself
- `TestClient` runs the FastAPI app in-process without starting a real server

**When would you use this approach instead of the smoke test?**
When your Docker image is very large (like this one at 8.67GB) and pulling it in CI takes too long. The trade-off is you test your API code but not the actual model. The smoke test tests the real model but takes longer.

---

## 10. .github/workflows/deploy.yml — The CI/CD Pipeline

This file defines the entire automated pipeline. GitHub reads it and runs it on every push to `main`.

### Trigger

```yaml
on:
  push:
    branches:
      - main
```

Only pushes to `main` trigger the pipeline. Always push to `main` — not `master`. Pushing to `master` goes to a different branch that nothing watches.

### Global Variable

```yaml
env:
  DOCKER_REPO: 03sarath/distilbert-sentiment
```

Docker Hub repository name — available to all jobs so it is defined once and not repeated.

---

### Job 1: smoke-test

```yaml
smoke-test:
  name: Model Smoke Test
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4

    - name: Start container
      run: |
        IMAGE=$(grep '^FROM' hf-space/Dockerfile | awk '{print $2}')
        docker run -d --name smoke-test -p 7860:7860 $IMAGE

    - name: Wait for model and run smoke test
      run: |
        curl --retry 20 --retry-delay 15 --retry-connrefused --retry-all-errors -sf \
          http://localhost:7860/health
        curl --fail -X POST http://localhost:7860/predict \
          -H "Content-Type: application/json" \
          -d '{"text": "The weather today is absolutely wonderful!"}'

    - name: Stop container
      if: always()
      run: docker stop smoke-test && docker rm smoke-test
```

**What each step does:**

**Step 1 — Start container:**
```bash
IMAGE=$(grep '^FROM' hf-space/Dockerfile | awk '{print $2}')
```
Reads `hf-space/Dockerfile`, extracts the image name (e.g., `03sarath/distilbert-sentiment:v1`). `grep '^FROM'` finds the FROM line, `awk '{print $2}'` extracts the second word (the image name).

```bash
docker run -d --name smoke-test -p 7860:7860 $IMAGE
```
Starts the container in the background (`-d`), names it `smoke-test`, maps port 7860. Docker automatically pulls the image from Docker Hub if not already present.

**Step 2 — Wait and test:**
```bash
curl --retry 20 --retry-delay 15 --retry-connrefused --retry-all-errors -sf \
  http://localhost:7860/health
```
Polls the `/health` endpoint until the model is ready. The two retry flags are both needed:
- `--retry-connrefused` — retries while the port is still closed (container starting)
- `--retry-all-errors` — retries while the port is open but not responding (model still downloading)

Without `--retry-all-errors`, curl exits with code 56 ("receive error") instead of retrying.

Retries every 15 seconds, up to 20 times (5 minutes maximum wait).

```bash
curl --fail -X POST http://localhost:7860/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "The weather today is absolutely wonderful!"}'
```
Sends a real sentence to the real DistilBERT model. `--fail` means any non-200 response fails the step and blocks deployment.

**Step 3 — Stop container:**
```bash
if: always()
```
This step runs even if previous steps failed — ensures the container is always cleaned up and no orphaned containers are left on the runner.

**Why this is better than mocked tests for MLOps:**
The smoke test validates the **actual artifact** being deployed — the real Docker image with the real model. If the image is broken, corrupted, or the model fails to load, the smoke test catches it before anything reaches Hugging Face Spaces.

**Time expectation:** 8–12 minutes (mostly pulling the 8.67GB image).

---

### Job 2: deploy

```yaml
deploy:
  name: Deploy to Hugging Face Spaces
  runs-on: ubuntu-latest
  needs: smoke-test
  steps:
    - name: Checkout repository
      uses: actions/checkout@v4

    - name: Upload to Hugging Face Space
      env:
        HF_TOKEN: ${{ secrets.HF_TOKEN }}
        HF_USERNAME: ${{ secrets.HF_USERNAME }}
        HF_SPACE_NAME: ${{ secrets.HF_SPACE_NAME }}
      run: |
        pip install huggingface_hub -q
        python -c "
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_folder(
            folder_path='hf-space/',
            repo_id='$HF_USERNAME/$HF_SPACE_NAME',
            repo_type='space',
            token='$HF_TOKEN'
        )
        print('Deployed successfully to HF Space')
        "
```

**What it does:**
Only runs after smoke-test passes (`needs: smoke-test`). Installs the `huggingface_hub` Python library and uses `HfApi.upload_folder()` to push the `hf-space/` folder (containing just the 2-line Dockerfile) to the Hugging Face Space git repository. HF Spaces detects the new commit, pulls the referenced Docker image, and runs it.

**Why `HfApi.upload_folder()` instead of `huggingface-cli`?**
The `huggingface-cli` command was deprecated. The Python `HfApi` is the underlying library that the CLI used — calling it directly is more stable and future-proof.

---

### Job 3: health-check

```yaml
health-check:
  name: Post-Deploy Health Check
  runs-on: ubuntu-latest
  needs: deploy
  steps:
    - name: Wait for Space to rebuild
      run: sleep 90

    - name: Call health endpoint and confirm version
      env:
        HF_USERNAME: ${{ secrets.HF_USERNAME }}
        HF_SPACE_NAME: ${{ secrets.HF_SPACE_NAME }}
      run: |
        curl --fail \
             --retry 5 \
             --retry-delay 30 \
             --retry-connrefused \
             https://$HF_USERNAME-$HF_SPACE_NAME.hf.space/health
```

**What it does:**
Waits 90 seconds for HF Spaces to pull the image and start the container, then calls the live `/health` endpoint on Hugging Face Spaces. Retries up to 5 times with 30-second gaps (up to 2.5 extra minutes). If the endpoint never responds successfully, the pipeline fails — meaning the deployment went through but the service is not healthy.

**The full job dependency chain:**
```
smoke-test  →  deploy  →  health-check
```

Each job only runs if the previous one passed. If smoke-test fails, nothing deploys.

---

## 11. Hugging Face Spaces — Where the Model Runs

**What is Hugging Face Spaces?**
Hugging Face Spaces is a free hosting platform for machine learning applications. It supports Gradio, Streamlit, and Docker containers.

**What type of Space does this project use?**
A **Docker Space**. HF gives you a git repository. You push a Dockerfile to it. HF builds and runs whatever that Dockerfile defines.

**How the Space works:**
```
Pipeline pushes hf-space/Dockerfile
         ↓
HF Spaces reads:  FROM 03sarath/distilbert-sentiment:v1
         ↓
HF Spaces pulls the image from Docker Hub
         ↓
HF Spaces runs the container
         ↓
Live at: https://03sarath-distilbert-sentiment.hf.space
```

**Is this running as a container?**
Yes. The entire application (FastAPI + DistilBERT) runs inside a Docker container on Hugging Face's infrastructure.

**Live URL format:**
```
https://{hf-username}-{space-name}.hf.space
```
This project: `https://03sarath-distilbert-sentiment.hf.space`

**Available endpoints:**

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Returns service status and version |
| `/predict` | POST | Accepts text, returns sentiment |
| `/docs` | GET | Auto-generated API documentation (FastAPI) |

---

## 12. Docker Hub — Where the Image Lives

**What is Docker Hub?**
A public registry for storing Docker images. It is to Docker images what GitHub is to code.

**Image naming convention:**
```
03sarath/distilbert-sentiment:v1
│         │                   │
│         │                   └── Tag (version)
│         └── Repository name
└── Docker Hub username
```

**Tags used in this project:**

| Tag | Size | Health response | Predict response |
|---|---|---|---|
| `latest` | 8.67GB | `{"status": "ok"}` | `{"label": "...", "score": ...}` |
| `v1` | 8.67GB | `{"status": "ok"}` | `{"label": "...", "score": ...}` |
| `v2` | 8.67GB | `{"status": "ok", "version": "v2"}` | `{"label": "...", "score": ..., "model_version": "v2"}` |

**How was the image built?**
Manually once on a developer's machine using Docker Desktop:
```bash
docker build -t 03sarath/distilbert-sentiment:latest ./app
docker push 03sarath/distilbert-sentiment:latest

docker tag 03sarath/distilbert-sentiment:latest 03sarath/distilbert-sentiment:v1
docker push 03sarath/distilbert-sentiment:v1
```

The pipeline never builds the image — it only pulls and references it.

---

## 13. GPU or CPU? What Hardware Is Used?

**Short answer: CPU only, no GPU.**

**On Hugging Face Spaces free tier:**
- CPU only
- 2 vCPUs, 16GB RAM
- No GPU

**Does DistilBERT need a GPU?**
No. DistilBERT is small enough to run on CPU for inference. It is slower than GPU but perfectly adequate for demos and low-to-medium traffic.

**Prediction latency on CPU:** approximately 50–200ms per request.

**If you needed GPU:**
HF Spaces offers paid GPU tiers (T4, A10G) configurable in Space settings. The code would not change — PyTorch automatically uses CUDA when a GPU is available.

**Why is the image 8.67GB if there is no GPU?**
The `torch==2.3.0` package bundles CUDA libraries inside itself (nvidia-cublas, nvidia-cudnn, etc.). These are included in the pip package regardless of whether a GPU is present. On a CPU-only machine, they are never loaded — PyTorch silently falls back to CPU execution.

---

## 14. Model Versioning — v1 vs v2

**What changed between v1 and v2:**

| | v1 | v2 |
|---|---|---|
| `MODEL_VERSION` constant | Not present | `"v2"` |
| `/health` response | `{"status": "ok"}` | `{"status": "ok", "version": "v2"}` |
| `/predict` response | `{"label": "...", "score": ...}` | `{"label": "...", "score": ..., "model_version": "v2"}` |
| `PredictResponse` schema | `label`, `score` | `label`, `score`, `model_version` |

**How to deploy v2 — one line change:**
```dockerfile
# hf-space/Dockerfile
FROM 03sarath/distilbert-sentiment:v2
```
`git commit` + `git push origin main` → pipeline runs automatically → v2 live.

**How to roll back to v1:**
```dockerfile
FROM 03sarath/distilbert-sentiment:v1
```
Push. Pipeline deploys v1 again. No server access needed.

**Why this is powerful:**
Every version is an immutable image on Docker Hub. You can deploy any version at any time by changing one word in one file and pushing. No rebuilding, no data loss, no manual steps.

---

## 15. GitHub Secrets — How Credentials Are Managed

Three secrets stored in GitHub (repo → Settings → Secrets and variables → Actions):

| Secret | Value | Used in |
|---|---|---|
| `HF_TOKEN` | Hugging Face write-access token | deploy job |
| `HF_USERNAME` | Hugging Face username | deploy job, health-check job |
| `HF_SPACE_NAME` | Name of the HF Space (`distilbert-sentiment`) | deploy job, health-check job |

**Why not put these in the workflow file?**
GitHub Secrets are encrypted at rest and only injected at pipeline runtime. They never appear in code or logs — shown as `***` in the Actions output. If you put credentials directly in the YAML file, anyone with repo access can see them.

**How they are used in the workflow:**
```yaml
env:
  HF_TOKEN: ${{ secrets.HF_TOKEN }}
```

---

## 16. End-to-End Flow: What Happens When You Push Code

Complete sequence from `git push` to a live updated model:

```
1. Developer changes hf-space/Dockerfile (e.g., v1 → v2)
2. git commit + git push origin main
3. GitHub detects push to main branch
4. GitHub Actions starts the workflow

5. JOB 1 (smoke-test)
   - Spins up Ubuntu VM
   - Downloads repo code
   - Reads hf-space/Dockerfile → extracts image name
   - docker run pulls image from Docker Hub (~8.67GB, takes 5-10 min)
   - Container starts, DistilBERT downloads from HF Hub (~250MB)
   - curl polls /health every 15s until model is ready
   - Sends real prediction → model responds
   - docker stop cleans up container
   - Smoke test passed → Job 1 complete

6. JOB 2 (deploy) — starts only after Job 1 passes
   - Spins up Ubuntu VM
   - Downloads repo code
   - pip install huggingface_hub
   - HfApi.upload_folder() pushes hf-space/Dockerfile to HF Space repo
   - HF Spaces detects the new commit
   - HF Spaces pulls the Docker image from Docker Hub
   - HF Spaces starts the container
   - Job 2 complete

7. JOB 3 (health-check) — starts after Job 2
   - Waits 90 seconds for HF Spaces to rebuild
   - curl calls GET https://03sarath-distilbert-sentiment.hf.space/health
   - Retries up to 5 times if not ready
   - Gets 200 OK → pipeline succeeds
   - If never responds → pipeline fails, team is alerted

8. Model is live at:
   https://03sarath-distilbert-sentiment.hf.space/predict
```

**Total time from push to live:** approximately 12–15 minutes (dominated by the 8.67GB image pull in smoke-test).

---

## 17. Key Decisions and Why

**Why not build the Docker image in the pipeline?**
The image is 8.67GB. Building it on every push would take 15+ minutes and waste compute. The image is built once manually when code changes, pushed to Docker Hub, and the pipeline just references it by tag.

**Why use `HfApi.upload_folder()` instead of git commands?**
The original pipeline used `git clone` + `git commit` + `git push` to update the HF Space. This required 5 steps and was fragile. The Python API does the same thing in 1 step and handles all the git operations internally.

**Why was `huggingface-cli` replaced?**
The `huggingface-cli` command was deprecated by Hugging Face. Running it produced a fatal error. The `HfApi` Python library is the stable, supported alternative.

**Why `--retry-all-errors` on the health check curl?**
Without it, curl exits with code 56 (receive error) when the container port is open but the HTTP server is not yet responding. `--retry-connrefused` alone only retries on connection refused (code 7) — it does not retry on receive errors. Both flags are needed to cover the full container startup sequence.

**Why keep `tests/test_app.py` if it is not used by the pipeline?**
It demonstrates an alternative CI approach (fast mocked tests vs slow real model test) and is useful for local development when you want to verify API logic without running the full container.

**Why push to `main` not `master`?**
The workflow trigger is `on: push: branches: main`. Pushing to `master` creates a separate branch that the pipeline ignores — a common source of confusion.

---

*This documentation reflects the final state of the project as built for the MLOps Specialization Course. The same pipeline structure applies to any ML model — replace the Docker Hub image and HF Space name to deploy a different model with zero pipeline changes.*
