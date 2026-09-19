#!/usr/bin/env bash
# Provision Azure Container Apps for the demo and roll out the current code.
# Safe to re-run: each run builds a new image tag and updates the app.
set -euo pipefail

RESOURCE_GROUP="${RESOURCE_GROUP:-rg-vihate}"
LOCATION="${LOCATION:-southeastasia}"
NAME_PREFIX="${NAME_PREFIX:-vihate}"
MIN_REPLICAS="${MIN_REPLICAS:-1}"
MODEL_ID="${MODEL_ID:-bvuong/visoBert-ensemble}"
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

# deploy <output-query> [extra bicep parameters...]: prints the single output value selected by the query.
deploy() {
  local query="$1"
  shift
  az deployment group create \
    --resource-group "$RESOURCE_GROUP" \
    --template-file "$template" \
    --parameters namePrefix="$NAME_PREFIX" minReplicas="$MIN_REPLICAS" location="$LOCATION" "$@" \
    --query "$query" --output tsv
}

echo "==> Registering resource providers (no-op if already registered)"
for ns in Microsoft.App Microsoft.OperationalInsights Microsoft.ContainerRegistry Microsoft.ManagedIdentity; do
  az provider register --namespace "$ns" --wait || echo "warning: could not register $ns (ARM will auto-register it during deployment)" >&2
done

echo "==> Resource group $RESOURCE_GROUP ($LOCATION)"
az group create --name "$RESOURCE_GROUP" --location "$LOCATION" --output none

echo "==> Shared infrastructure (registry, identity, environment)"
registry="$(deploy "properties.outputs.registryName.value")"

echo "==> Building image $NAME_PREFIX:$IMAGE_TAG in $registry"
az acr build \
  --registry "$registry" \
  --image "$NAME_PREFIX:$IMAGE_TAG" \
  --build-arg MODEL_ID="$MODEL_ID" \
  "$repo_root"

echo "==> Deploying the app"
fqdn="$(deploy "properties.outputs.appUrl.value" imageTag="$IMAGE_TAG")"

echo
echo "Deployed: https://$fqdn"
