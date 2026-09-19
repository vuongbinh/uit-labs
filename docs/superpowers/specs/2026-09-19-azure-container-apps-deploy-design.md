# Azure Container Apps deployment: design

## Goal

Make the Gradio demo (`bvuong/visoBert-ensemble`) deployable to Azure Container Apps with one
repeatable command, using Bicep for infrastructure and an `az` CLI script for orchestration.
The Hugging Face Space deployment keeps working unchanged.

## Non-goals

- CI/CD (GitHub Actions). Deploys are run by hand.
- App authentication. It is a public demo, like the Space.
- GPU, custom domains, VNet integration, autoscaling tuning beyond the defaults below.

## Files

| File | Change |
| --- | --- |
| `Dockerfile` | New. Multi-stage build, model baked in. |
| `.dockerignore` | New. Keeps the build context small. |
| `infra/main.bicep` | New. All Azure resources. |
| `infra/deploy.sh` | New. Provision, build in ACR, roll out. |
| `README.md` | New "Deploy to Azure Container Apps" section. |
| `app.py`, `ui.py`, `model.py`, `batch.py` | Unchanged. |

## Container image

Multi-stage build.

**Builder** (`python:3.12-slim` + `uv`):

- `uv sync --frozen --no-dev` installs the exact versions in `uv.lock`. Torch comes from the
  CPU index already configured in `pyproject.toml`.
- Downloads the model and tokenizer into `HF_HOME=/opt/hf` by calling the app's own loader
  (`model.Classifier`), so the cached files are exactly what runtime loads. The model id comes
  from a build arg defaulting to `bvuong/visoBert-ensemble`.

**Runtime** (`python:3.12-slim`):

- Copies `.venv`, `/opt/hf`, and `app.py ui.py model.py batch.py`.
- Runs as a non-root user.
- Env: `HF_HOME=/opt/hf`, `HF_HUB_OFFLINE=1`, `GRADIO_SERVER_NAME=0.0.0.0`,
  `GRADIO_SERVER_PORT=7860`, `PYTHONUNBUFFERED=1`. Gradio reads the two `GRADIO_*` variables
  itself, so `app.py` needs no change and local/Space launches are unaffected.
- `EXPOSE 7860`, `CMD ["python", "app.py"]`.

Consequence of `HF_HUB_OFFLINE=1`: setting `MODEL_ID` at runtime only works for a checkpoint
already in the image. Serving a different model means rebuilding with `--build-arg MODEL_ID=...`.

## Infrastructure (`infra/main.bicep`)

Resource-group scope. Resources:

1. Log Analytics workspace (Container Apps logs).
2. Azure Container Registry, Basic SKU, admin user disabled.
3. User-assigned managed identity with `AcrPull` on the registry.
4. Container Apps managed environment, wired to the workspace.
5. Container App using the identity to pull from ACR:
   - External ingress, target port 7860, HTTPS only.
   - 2 vCPU / 4 GiB (torch + transformer OOMs at the 1 GiB default).
   - Min replicas parameter (default 1), max replicas 3, HTTP concurrency scale rule.
   - Startup and liveness probes on `GET /`.

Parameters: `location` (default `southeastasia`), `namePrefix` (default `vihate`), `imageTag` (empty means deploy only the shared infrastructure, without the app), `minReplicas` (default 1).
Output: the app FQDN and the registry login server.

No secrets exist anywhere: the model is public and the registry pull uses managed identity.

## Deploy script (`infra/deploy.sh`)

`set -euo pipefail`. Inputs via env vars with defaults: `RESOURCE_GROUP` (`rg-vihate`),
`LOCATION` (`southeastasia`), `IMAGE_TAG` (UTC timestamp).

1. `az group create`.
2. Deploy Bicep with an empty image tag, which creates the registry, identity and environment but no app.
3. `az acr build` the image remotely (no local Docker required).
4. Deploy Bicep again with the real image tag, which creates or updates the app. The role assignment already exists, so the pull succeeds.
5. Print the app URL.

Idempotent: re-running with a new tag ships a new version. Fails fast if `az` is missing or not
logged in, with a clear message.

## README

New section covering prerequisites (just `az login`), running
`infra/deploy.sh`, overriding settings (`MIN_REPLICAS=0` for scale-to-zero, with the cold-start
tradeoff noted), rebuilding for a different model, and tear-down via `az group delete`.

## Verification

- `docker build` succeeds and the image runs locally: `curl localhost:7860` returns 200 and a
  prediction returns a label.
- The image contains the model cache and starts with no network (`HF_HUB_OFFLINE=1`).
- `az bicep build infra/main.bicep` compiles cleanly, and `shellcheck infra/deploy.sh` is clean
  if available.
- Existing `pytest` and `ruff check` still pass.

None of the checks above has been run yet: the authoring environment had no Docker daemon, `az` or `bicep`. The first real build and `infra/deploy.sh` run is the acceptance test.

Not verifiable here: an actual Azure deployment (no `az` CLI or subscription in this
environment). The first real run is the acceptance test.

## Risks

- **Image size:** CPU torch plus weights is roughly 2–3 GB. Acceptable for ACR Basic; build
  times are longer on first run because of the model download.
- **Cold start:** with `minReplicas=0`, the first request waits for image pull and model load.
  Default of 1 avoids this at a small ongoing cost.
- **Bootstrap ordering:** the app cannot reference an image until ACR exists, and a placeholder image would fail the port-7860 probes, so the first phase omits the app entirely.
