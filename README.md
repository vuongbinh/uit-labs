---
title: Vietnamese Hate Speech Detection
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.50.0
python_version: "3.12"
app_file: app.py
pinned: false
---

# Vietnamese Hate Speech Detection

Gradio demo for [`bvuong/nlp-vihate`](https://huggingface.co/bvuong/nlp-vihate): classifies
Vietnamese text as `CLEAN`, `OFFENSIVE` or `HATE`, one comment at a time or from a
`.txt`/`.csv`/`.tsv` file.

## Run locally

```bash
uv sync
uv run app.py                       # plain launch, http://127.0.0.1:7860
uv run gradio app.py                # hot-reloading dev server
MODEL_ID=path/or/repo uv run app.py # serve a different checkpoint
uv run pytest                       # tests (no downloads)
```

## Layout

| File | Job |
| --- | --- |
| `app.py` | Entrypoint: loads the model, exposes `demo` |
| `ui.py` | Gradio layout and handlers |
| `model.py` | Load the checkpoint, `predict(texts)` |
| `batch.py` | Parse uploaded files, build result tables |

To add a feature, add a tab in `ui.py`; model changes stay in `model.py`.

## Deploy to a Hugging Face Space

Gradio Spaces install only from `requirements.txt`, so generate it from the lockfile
(it is gitignored and never edited by hand):

```bash
{ echo "--extra-index-url https://download.pytorch.org/whl/cpu"; uv export --no-hashes --no-dev --no-emit-project; } > requirements.txt
```

The first line points pip at the PyTorch CPU wheel index, because `uv export` omits the explicit index.

A Space is its own git repo, so copy the files into a clone of it and push from there:

```bash
git clone https://huggingface.co/spaces/<user>/<space>
cp app.py ui.py model.py batch.py requirements.txt README.md <space-clone>/
cd <space-clone> && git add -A && git commit -m "Update demo" && git push
```

`requirements.txt` is intentionally gitignored here, so never rely on pushing this repo directly.
Create the Space with SDK **Gradio**. `bvuong/nlp-vihate` is public, so no token is needed; the
first start downloads the weights and later starts use the cache. The sdk version in the header above
must match the `gradio` version in `uv.lock`.

## Deploy to Azure Container Apps

The `Dockerfile` builds a CPU-only image with the model weights baked in, and `infra/` provisions
everything else (Log Analytics, Container Registry, managed identity, Container Apps environment
and app). Requires the [Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli) and
an Azure subscription; no local Docker is needed because the image is built in the registry.

```bash
az login
infra/deploy.sh          # prints https://<app>.<environment-id>.<region>.azurecontainerapps.io when done
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

Every deploy pushes a new ~2-3 GB image tag and the Basic registry keeps them all, so delete old tags with `az acr repository delete --name <registry> --image vihate:<tag>` when storage matters.

To delete everything: `az group delete --name "${RESOURCE_GROUP:-rg-vihate}"`.
