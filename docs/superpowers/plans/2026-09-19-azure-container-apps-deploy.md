# Azure Container Apps Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Gradio demo deployable to Azure Container Apps with one repeatable command.

**Architecture:** A multi-stage Dockerfile builds a CPU-only image with the model weights baked in. `infra/main.bicep` declares the Azure resources (Log Analytics, ACR, managed identity with `AcrPull`, Container Apps environment, Container App). `infra/deploy.sh` runs it in two phases: infra without the app, then a remote `az acr build`, then the app with the new image tag.

**Tech Stack:** Docker, uv, Python 3.12, Bicep, Azure CLI (`az`), Azure Container Registry, Azure Container Apps.

**Spec:** `docs/superpowers/specs/2026-09-19-azure-container-apps-deploy-design.md`

**Deviations from spec (deliberate; Task 4 updates the spec to match):**

1. No placeholder image. The spec's bootstrap deploy used a public placeholder image, but the app has a startup probe on port 7860 and any placeholder image would fail it, so the first deploy would never turn healthy. Instead the Bicep app resource is conditional: `imageTag` empty means "infra only, no app". This replaces the spec's `containerImage` parameter.
2. Default `IMAGE_TAG` is a timestamp, not the git SHA. Rebuilding an unchanged SHA from a dirty tree would overwrite the tag without changing the template, so no new revision would roll out.
3. The `containerapp` az extension is not needed (Bicep and `az acr build` are core commands), so the README does not mention it.

## Global Constraints

- App code (`app.py`, `ui.py`, `model.py`, `batch.py`) is unchanged. Container binding comes from `GRADIO_SERVER_NAME=0.0.0.0` and `GRADIO_SERVER_PORT=7860` env vars.
- Python 3.12; dependencies come from `uv.lock` via `uv sync --frozen --no-dev` (CPU torch index is already in `pyproject.toml`).
- Model default `bvuong/nlp-vihate`, baked in at build time; runtime uses `HF_HUB_OFFLINE=1`.
- Container runs as a non-root user; port 7860; 2 vCPU / 4 GiB; max replicas 3; default min replicas 1.
- No secrets: registry pull uses a user-assigned managed identity, ACR admin user disabled.
- Default region `southeastasia`, default resource group `rg-vihate`, default `namePrefix` `vihate`.
- Do not touch the unrelated working-tree changes (`data/` deletions, `cyber_bullying/` notebook deletion). Stage files by explicit path only.
- Commit messages end with `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`.
- Ruff is `select = ["ALL"]`; `pytest` and `ruff check` must still pass at the end.

---

### Task 1: Container image

**Files:**
- Create: `Dockerfile`
- Create: `.dockerignore`

**Interfaces:**
- Produces: an image serving Gradio on `0.0.0.0:7860`, with the build arg `MODEL_ID` (default `bvuong/nlp-vihate`). Task 3 (`az acr build ... --build-arg MODEL_ID=...`) and Task 4 (README) rely on the arg name `MODEL_ID`.

- [ ] **Step 1: Pin the uv version used in the image**

Run: `uv --version`
Expected: something like `uv 0.8.x`. Use that `major.minor` as `UV_VERSION` in the Dockerfile below (replace `0.8` if yours differs).

- [ ] **Step 2: Write `.dockerignore`**

```gitignore
.git
.github
.venv
.codegraph
.omc
.gradio
.pytest_cache
.ruff_cache
__pycache__
*.py[cod]
data
docs
notebooks
cyber_bullying
src
tests
infra
outputs
requirements.txt
README.md
```

- [ ] **Step 3: Write `Dockerfile`**

```dockerfile
# syntax=docker/dockerfile:1

# ---- builder: resolve deps from the lockfile and bake the model into the HF cache ----
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0
WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Load the model through the app's own loader so the cache holds exactly what runtime reads.
ARG MODEL_ID=bvuong/nlp-vihate
ENV HF_HOME=/opt/hf
COPY model.py ./
RUN MODEL_ID="$MODEL_ID" .venv/bin/python -c "from model import Classifier; Classifier()"

# ---- runtime: no build tooling, no network needed ----
FROM python:3.12-slim
ARG MODEL_ID=bvuong/nlp-vihate
ENV MODEL_ID=$MODEL_ID \
    HF_HOME=/opt/hf \
    HF_HUB_OFFLINE=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    GRADIO_ANALYTICS_ENABLED=False \
    PYTHONUNBUFFERED=1 \
    PATH=/app/.venv/bin:$PATH

RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=builder --chown=app:app /opt/hf /opt/hf
COPY --chown=app:app app.py ui.py model.py batch.py ./

USER app
EXPOSE 7860
CMD ["python", "app.py"]
```

- [ ] **Step 4: Build the image**

Run: `docker build -t vihate-demo:local .`
Expected: succeeds. The builder stage downloads the model (visible as a "Downloading" step in the log). If the `uv:0.8` tag does not exist, use the version from Step 1.

- [ ] **Step 5: Smoke test: server answers on the port**

```bash
docker run --rm -d -p 7860:7860 --name vihate-smoke vihate-demo:local
until curl -fsS -o /dev/null http://localhost:7860/; do sleep 2; done
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:7860/
```

Expected: `200`. If the loop never ends, run `docker logs vihate-smoke` and fix (typical causes: server bound to 127.0.0.1, so the env vars did not apply; or model load error).

- [ ] **Step 6: Verify the weights are baked in and the container needs no network**

```bash
docker run --rm --network none vihate-demo:local \
  python -c "from model import Classifier; print(Classifier().predict(['xin chào']))"
```

Expected: a list with one dict of `CLEAN`/`OFFENSIVE`/`HATE` probabilities, no download attempt or connection error.

- [ ] **Step 7: Check user and size, then clean up**

```bash
docker exec vihate-smoke id -u
docker images vihate-demo:local --format '{{.Size}}'
docker stop vihate-smoke
```

Expected: `1000` (not root); image size roughly 2–3 GB. If it is far larger, check that torch installed from the CPU index (`docker run --rm vihate-demo:local python -c "import torch; print(torch.__version__)"` should end in `+cpu`).

- [ ] **Step 8: Commit**

```bash
git add Dockerfile .dockerignore
git commit -m "feat: add container image with baked-in model

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 2: Bicep infrastructure

**Files:**
- Create: `infra/main.bicep`

**Interfaces:**
- Consumes: image name convention `<acr login server>/<namePrefix>:<imageTag>` (Task 3 builds exactly this).
- Produces: parameters `location`, `namePrefix`, `imageTag` (empty = skip the app), `minReplicas`; outputs `registryName` (string), `registryLoginServer` (string), `appUrl` (string, empty when the app is not deployed). Task 3 reads `registryName` and `appUrl` by these names.

- [ ] **Step 1: Write `infra/main.bicep`**

```bicep
targetScope = 'resourceGroup'

@description('Azure region for all resources.')
param location string = resourceGroup().location

@description('Prefix for resource names and the image repository. Lowercase letters and digits only.')
@minLength(3)
@maxLength(12)
param namePrefix string = 'vihate'

@description('Image tag in the registry. Leave empty to deploy only the shared infrastructure (no app yet).')
param imageTag string = ''

@description('Minimum replicas. 0 enables scale-to-zero (cold starts load the model again).')
@minValue(0)
@maxValue(3)
param minReplicas int = 1

var registryName = '${namePrefix}${uniqueString(resourceGroup().id)}'
var acrPullRoleId = '7f951dda-4ed3-4680-a7ca-43fe172d538d'
var appPort = 7860

resource workspace 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: '${namePrefix}-logs'
  location: location
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: 30
  }
}

resource registry 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: registryName
  location: location
  sku: { name: 'Basic' }
  properties: {
    adminUserEnabled: false
  }
}

resource identity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: '${namePrefix}-id'
  location: location
}

resource acrPull 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(registry.id, identity.id, acrPullRoleId)
  scope: registry
  properties: {
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', acrPullRoleId)
  }
}

resource environment 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: '${namePrefix}-env'
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: workspace.properties.customerId
        sharedKey: workspace.listKeys().primarySharedKey
      }
    }
  }
}

resource app 'Microsoft.App/containerApps@2024-03-01' = if (!empty(imageTag)) {
  name: '${namePrefix}-app'
  location: location
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: { '${identity.id}': {} }
  }
  dependsOn: [acrPull]
  properties: {
    managedEnvironmentId: environment.id
    configuration: {
      ingress: {
        external: true
        targetPort: appPort
        transport: 'auto'
        allowInsecure: false
      }
      registries: [
        {
          server: registry.properties.loginServer
          identity: identity.id
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'app'
          image: '${registry.properties.loginServer}/${namePrefix}:${imageTag}'
          resources: {
            cpu: json('2')
            memory: '4Gi'
          }
          probes: [
            {
              type: 'Startup'
              httpGet: { path: '/', port: appPort }
              initialDelaySeconds: 10
              periodSeconds: 10
              failureThreshold: 10
            }
            {
              type: 'Liveness'
              httpGet: { path: '/', port: appPort }
              periodSeconds: 30
              failureThreshold: 3
            }
          ]
        }
      ]
      scale: {
        minReplicas: minReplicas
        maxReplicas: 3
        rules: [
          {
            name: 'http-concurrency'
            http: { metadata: { concurrentRequests: '10' } }
          }
        ]
      }
    }
  }
}

output registryName string = registry.name
output registryLoginServer string = registry.properties.loginServer
output appUrl string = app.?properties.configuration.ingress.fqdn ?? ''
```

Notes for the implementer: Container Apps limits probe `failureThreshold` to 10 and `initialDelaySeconds` to 60, which is why the startup probe is 10s + 10 x 10s (about 110 s to load the model). `appUrl` returns the bare FQDN; the script prefixes `https://`.

- [ ] **Step 2: Compile the template**

Run: `az bicep build --file infra/main.bicep --stdout > /dev/null`
Expected: no errors. Warnings about `listKeys` are acceptable; anything else is not.

If `az` is not installed (it is not on this machine as of writing), **ask the user before installing anything**. Options to offer: `brew install azure-cli`, or the standalone Bicep compiler (`brew install bicep`, then `bicep build infra/main.bicep --stdout > /dev/null`). If the user declines both, say plainly that the template is uncompiled and skip to Step 3.

- [ ] **Step 3: Commit**

```bash
git add infra/main.bicep
git commit -m "feat: add Bicep template for Azure Container Apps

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 3: Deploy script

**Files:**
- Create: `infra/deploy.sh` (executable)

**Interfaces:**
- Consumes: from `infra/main.bicep`, parameters `location`, `namePrefix`, `imageTag`, `minReplicas`, outputs `registryName` and `appUrl`; from the `Dockerfile`, build arg `MODEL_ID`. The image repository name equals `namePrefix`.
- Produces: env-var interface `RESOURCE_GROUP` (default `rg-vihate`), `LOCATION` (default `southeastasia`), `NAME_PREFIX` (default `vihate`), `MIN_REPLICAS` (default `1`), `MODEL_ID` (default `bvuong/nlp-vihate`), `IMAGE_TAG` (default UTC timestamp). Task 4 documents these exact names.

- [ ] **Step 1: Write `infra/deploy.sh`**

```bash
#!/usr/bin/env bash
# Provision Azure Container Apps for the demo and roll out the current code.
# Safe to re-run: each run builds a new image tag and updates the app.
set -euo pipefail

RESOURCE_GROUP="${RESOURCE_GROUP:-rg-vihate}"
LOCATION="${LOCATION:-southeastasia}"
NAME_PREFIX="${NAME_PREFIX:-vihate}"
MIN_REPLICAS="${MIN_REPLICAS:-1}"
MODEL_ID="${MODEL_ID:-bvuong/nlp-vihate}"
IMAGE_TAG="${IMAGE_TAG:-$(date -u +%Y%m%d%H%M%S)}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
template="$repo_root/infra/main.bicep"

if ! command -v az >/dev/null 2>&1; then
  echo "error: the Azure CLI (az) is not installed: https://learn.microsoft.com/cli/azure/install-azure-cli" >&2
  exit 1
fi
if ! az account show >/dev/null 2>&1; then
  echo "error: not logged in to Azure; run 'az login' first" >&2
  exit 1
fi

deploy() {
  az deployment group create \
    --resource-group "$RESOURCE_GROUP" \
    --template-file "$template" \
    --parameters namePrefix="$NAME_PREFIX" minReplicas="$MIN_REPLICAS" location="$LOCATION" "$@" \
    --query "properties.outputs" --output json
}

echo "==> Registering resource providers (no-op if already registered)"
az provider register --namespace Microsoft.App --wait
az provider register --namespace Microsoft.OperationalInsights --wait
az provider register --namespace Microsoft.ContainerRegistry --wait

echo "==> Resource group $RESOURCE_GROUP ($LOCATION)"
az group create --name "$RESOURCE_GROUP" --location "$LOCATION" --output none

echo "==> Shared infrastructure (registry, identity, environment)"
registry="$(deploy | python3 -c 'import json,sys; print(json.load(sys.stdin)["registryName"]["value"])')"

echo "==> Building image $NAME_PREFIX:$IMAGE_TAG in $registry"
az acr build \
  --registry "$registry" \
  --image "$NAME_PREFIX:$IMAGE_TAG" \
  --build-arg MODEL_ID="$MODEL_ID" \
  "$repo_root"

echo "==> Deploying the app"
fqdn="$(deploy imageTag="$IMAGE_TAG" | python3 -c 'import json,sys; print(json.load(sys.stdin)["appUrl"]["value"])')"

echo
echo "Deployed: https://$fqdn"
```

Notes for the implementer: it parses the deployment outputs with `python3` (always present here) rather than adding a `jq` dependency. `--query "properties.outputs"` makes `az` print only the outputs object.

- [ ] **Step 2: Make it executable and syntax-check it**

```bash
chmod +x infra/deploy.sh
bash -n infra/deploy.sh
```

Expected: no output.

- [ ] **Step 3: Lint with shellcheck if available**

Run: `command -v shellcheck && shellcheck infra/deploy.sh || echo "shellcheck not installed, skipped"`
Expected: no findings, or the "skipped" message.

- [ ] **Step 4: Verify the fail-fast path**

Run: `PATH=/usr/bin:/bin bash infra/deploy.sh; echo "exit=$?"`
Expected (if `az` is not in `/usr/bin` or `/bin`): the "Azure CLI (az) is not installed" message and `exit=1`. If `az` is there, skip this step.

- [ ] **Step 5: Commit**

```bash
git add infra/deploy.sh
git commit -m "feat: add deploy script for Azure Container Apps

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 4: Documentation and spec sync

**Files:**
- Modify: `README.md` (append a section after "Deploy to a Hugging Face Space")
- Modify: `docs/superpowers/specs/2026-09-19-azure-container-apps-deploy-design.md`

**Interfaces:**
- Consumes: the env-var names from Task 3 and the build arg from Task 1.

- [ ] **Step 1: Append the README section**

Add this at the end of `README.md`:

````markdown

## Deploy to Azure Container Apps

The `Dockerfile` builds a CPU-only image with the model weights baked in, and `infra/` provisions
everything else (Log Analytics, Container Registry, managed identity, Container Apps environment
and app). Requires the [Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli) and
an Azure subscription; no local Docker is needed because the image is built in the registry.

```bash
az login
infra/deploy.sh          # prints https://<app>.<region>.azurecontainerapps.io when done
```

Re-running the script builds a new image tag and rolls out a new revision. Settings are env vars:

| Variable | Default | Meaning |
| --- | --- | --- |
| `RESOURCE_GROUP` | `rg-vihate` | Resource group to create/use |
| `LOCATION` | `southeastasia` | Azure region |
| `NAME_PREFIX` | `vihate` | Prefix for resource names and the image repository |
| `MIN_REPLICAS` | `1` | `0` scales to zero and saves money, but the first request after idle waits for the model to load |
| `MODEL_ID` | `bvuong/nlp-vihate` | Checkpoint baked into the image |
| `IMAGE_TAG` | UTC timestamp | Tag for this build |

The app runs with 2 vCPU / 4 GiB (a smaller size runs out of memory loading the model) and scales
to 3 replicas. The container starts offline (`HF_HUB_OFFLINE=1`), so `MODEL_ID` is fixed at build
time; to serve another checkpoint, re-run the script with a different `MODEL_ID`.

To try the image locally: `docker build -t vihate-demo . && docker run --rm -p 7860:7860 vihate-demo`.

To delete everything: `az group delete --name rg-vihate`.
````

- [ ] **Step 2: Bring the spec in line with the plan's deviations**

In `docs/superpowers/specs/2026-09-19-azure-container-apps-deploy-design.md`:

- In "Infrastructure", replace the parameters sentence with: ``Parameters: `location` (default `southeastasia`), `namePrefix` (default `vihate`), `imageTag` (empty means deploy only the shared infrastructure, without the app), `minReplicas` (default 1).``
- In "Deploy script", replace step 2 with: `2. Deploy Bicep with an empty image tag, which creates the registry, identity and environment but no app.` and step 4 with: `4. Deploy Bicep again with the real image tag, which creates or updates the app. The role assignment already exists, so the pull succeeds.`
- In "Deploy script" inputs, change `IMAGE_TAG` (git short SHA, else timestamp) to `IMAGE_TAG` (UTC timestamp).
- In "Risks", replace the "Bootstrap ordering" bullet with: `- **Bootstrap ordering:** the app cannot reference an image until ACR exists, and a placeholder image would fail the port-7860 probes, so the first phase omits the app entirely.`
- In "README", delete "with the `containerapp` extension" wording: the prerequisites are just `az login`.
- Fix markdownlint MD032 (lists need blank lines around them) at the two places the linter flagged, the lists under "**Builder**" and "**Runtime**" in "Container image": add a blank line between each bold heading line and the list that follows.

- [ ] **Step 3: Confirm nothing regressed**

Run: `uv run pytest && uv run ruff check`
Expected: tests pass, `All checks passed!`.

- [ ] **Step 4: Commit**

```bash
git add README.md docs/superpowers/specs/2026-09-19-azure-container-apps-deploy-design.md docs/superpowers/plans/2026-09-19-azure-container-apps-deploy.md
git commit -m "docs: document Azure Container Apps deployment

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Acceptance (after all tasks)

Not verifiable in this environment: an actual Azure deployment (no `az` CLI or subscription). The first real `infra/deploy.sh` run is the acceptance test: it should print an `https://` URL that serves the UI and returns a prediction. Things most likely to need a tweak on that first run: probe timing if the model loads slower than about 110 s, and the `uv` image tag if the pinned minor version is unavailable.
