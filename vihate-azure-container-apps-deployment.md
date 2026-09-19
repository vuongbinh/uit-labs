# ViHate Deployment — Azure Container Apps + GitHub Actions

Deploy the ViHate Vietnamese hate-speech demo (Gradio) to a public HTTPS URL with no Hugging Face or Azure login for users, 2 GiB RAM, and low idle cost. Development is in WSL; images are built remotely by GitHub Actions.

```text
WSL --git push--> GitHub --Actions: docker build/push--> GHCR (public)
                                                           |
                                              anonymous pull
                                                           v
                                     Azure Container Apps --> public URL
```

**Why not a Hugging Face Space:** embedding it in an iframe doesn't change its 512 MiB RAM limit or ZeroGPU quota/auth.

**Baseline**

| Item | Value |
|---|---|
| Resource group | `rg-vihate` |
| Environment | `vihate-env` (Express, reused) |
| Region | `eastasia` |
| App | `vihate` |
| Runtime | 1 vCPU, 2 GiB, min 0 / max 1 replicas, external ingress, port 7860 |
| Image | `ghcr.io/<github-user-or-org>/vihate:latest` |

---

## Part 1 — App and container

**`app.py`** must bind to all interfaces:

```python
import os

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
```

**Dockerfile** (uses `uv`; adjust `CMD` if the entrypoint differs):

```dockerfile
FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY . .
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PORT=7860
EXPOSE 7860
CMD ["python", "app.py"]
```

**`.dockerignore`** (don't ignore the model directory if it is bundled):

```text
.git
.github
.venv
__pycache__
*.pyc
.pytest_cache
.idea
.vscode
.env
```

**Model:** either download from Hugging Face at startup (smaller image, slower cold start) or bake it into the image and load from `/app/model` (no runtime download). Use `device = "cuda" if torch.cuda.is_available() else "cpu"`. Never hardcode `cuda`.

---

## Part 2 — GitHub: build and push to GHCR

Create `.github/workflows/build-vihate.yml`:

```yaml
name: Build ViHate Docker Image

on:
  push:
    branches:
      - nlp/vihsd-demo-persistence
  workflow_dispatch:

permissions:
  contents: read
  packages: write

jobs:
  docker:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - uses: docker/setup-buildx-action@v3
      - uses: docker/build-push-action@v6
        with:
          context: .
          push: true
          tags: |
            ghcr.io/${{ github.repository_owner }}/vihate:latest
            ghcr.io/${{ github.repository_owner }}/vihate:${{ github.sha }}
```

`GITHUB_TOKEN` is provided automatically; no manual token is needed.

```bash
git add Dockerfile .dockerignore .github/workflows/build-vihate.yml
git commit -m "Add ViHate container deployment"
git push
```

**Make the package public** after the first successful run: GitHub → Profile/Org → Packages → `vihate` → Package settings → Change visibility → Public. Azure Express then pulls anonymously, with no registry credentials or managed identity.

---

## Part 3 — Azure setup

### 3.1 CLI and subscription

```bash
az login
az account list -o table
az account set --subscription "<SUBSCRIPTION_ID>"
az extension add --name containerapp --upgrade
az provider register --namespace Microsoft.App
az provider register --namespace Microsoft.OperationalInsights
```

### 3.2 Check allowed regions (Azure for Students)

Don't guess the region. Read the subscription policy first:

```bash
SUB_ID=$(az account show --query id -o tsv)
az policy assignment list \
  --scope "/subscriptions/$SUB_ID" \
  --disable-scope-strict-match true \
  --query "[].{Name:displayName,Parameters:parameters}" -o jsonc
```

Allowed here: `eastasia`, `malaysiawest`, `indonesiacentral`, `koreacentral`, `japanwest`. This project uses `eastasia`.

### 3.3 Resource group and environment

```bash
az group create --name rg-vihate --location eastasia
```

The Student subscription allows only **one** Container Apps environment per region, so reuse `vihate-env`:

```bash
az containerapp env list -o table
az containerapp env show --name vihate-env --resource-group rg-vihate -o jsonc
```

If you must create one, skip Log Analytics (it can be auto-created in a disallowed region):

```bash
az containerapp env create \
  --name vihate-env --resource-group rg-vihate \
  --location eastasia --logs-destination none
```

### 3.4 Create the app

Run after the GHCR package is public. Do **not** pass `--source`, `--registry-*`, `--registry-identity` or `--system-assigned`.

```bash
az containerapp create \
  --name vihate \
  --resource-group rg-vihate \
  --environment vihate-env \
  --image ghcr.io/<github-user-or-org>/vihate:latest \
  --ingress external \
  --target-port 7860 \
  --cpu 1.0 \
  --memory 2.0Gi \
  --min-replicas 0 \
  --max-replicas 1
```

### 3.5 Public URL

```bash
az containerapp show --name vihate --resource-group rg-vihate \
  --query properties.configuration.ingress.fqdn -o tsv
```

Open `https://<returned-fqdn>`.

---

## Part 4 — Operate

**Update after a new image is pushed** (prefer a commit-SHA tag over `latest` for reproducibility):

```bash
az containerapp update \
  --name vihate --resource-group rg-vihate \
  --image ghcr.io/<github-user-or-org>/vihate:<git-sha>
```

**Status and logs:**

```bash
az containerapp list --resource-group rg-vihate -o table
az containerapp revision list --name vihate --resource-group rg-vihate -o table
az containerapp logs show --name vihate --resource-group rg-vihate --follow
```

Look for `ModuleNotFoundError`, `Killed`/`OutOfMemory`, model download failures, and port/ingress mismatch.

**Delete a broken app** (normal during redeploys; never delete the environment):

```bash
az containerapp delete --name vihate --resource-group rg-vihate --yes
```

**Next steps, once stable:** add `az containerapp update` as a final workflow step for automatic revisions; consider ONNX + INT8 quantization on ONNX Runtime CPU if RAM or cold start becomes a problem.

---

## Issues and solutions

| Error | Cause | Solution |
|---|---|---|
| `Out of memory (used over 512Mi)` | Hugging Face Space RAM limit | Move to Container Apps with 2 GiB |
| `RequestDisallowedByAzure` | Region blocked by the Student policy (also hit when Azure auto-created a Log Analytics workspace) | Check allowed regions (3.2), deploy to one, and use `--logs-destination none` |
| `MaxNumberOfRegionalEnvironmentsInSubExceeded` | Only 1 environment allowed per region | Reuse `vihate-env` |
| Existing environment can't be converted to Consumption-only | Environment type can't be changed after creation | Reuse it as is |
| `ManagedEnvironmentHasContainerApps` | Tried to delete an environment that still has apps | Delete the app first |
| `ExpressEnvironmentFeatureNotSupported` (system-assigned identity for registry auth) | `az containerapp up --source .` triggers Azure's build and registry flow, which Express doesn't support | Build in GitHub Actions, push to a public GHCR image, and use `az containerapp create --image ...` |
| `failed to connect to the docker API at unix:///var/run/docker.sock` | No Docker daemon in WSL | Build in GitHub Actions instead of locally |
| App unreachable or failing health check | Gradio bound to `127.0.0.1`, or port mismatch | Bind `0.0.0.0` and match `--target-port` to `PORT` (7860) |
| Partially-created broken app after a failed deploy | Failed deploys can still create the resource | `az containerapp delete`, then recreate |
