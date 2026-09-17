"""Persisted demo-bundle contract: what Part 7B writes and the demo reads back.

A *demo bundle* is one directory holding everything the live demo needs and
nothing it does not::

    <bundle>/
      model/            weights + tokenizer, as save_pretrained() wrote them
      demo_config.json  labels, calibrated t*, max_length, training + test metrics
      README.md         model card (this is what the Hub renders)
      demo_test_set.csv optional batch-tab sample (local bundles only)

Persisting the threshold alongside the weights is what makes a training-free demo
possible: t* costs about an hour of cross-validation to compute in Part 6, so a
bundle without it would still need the training run to reproduce it.

Stdlib plus `vihate.reporting` only, so this imports before the heavy ML
dependencies are needed. `resolve_bundle` imports huggingface_hub lazily, and only
when handed a Hub repo id instead of a local path.
"""

import json
import shutil
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from vihate.reporting import JsonValue, write_json

SCHEMA_VERSION: Final = 1
CONFIG_NAME: Final = "demo_config.json"
MODEL_DIRNAME: Final = "model"
CARD_NAME: Final = "README.md"
SAMPLE_NAME: Final = "demo_test_set.csv"
TASK_ID: Final = "vihsd-3way"
DEFAULT_REPO_HINT: Final = "<your-hf-user>/vihsd-visobert"
REQUIRED_KEYS: Final = ("labels", "backbone", "max_length", "threshold")


class DemoBundleError(Exception):
    """Raised when a directory or Hub repo id is not a usable demo bundle."""

    def __init__(self, reason: str, *, source: str | Path | None = None) -> None:
        """Build a message that names the offending source and what is wrong with it."""
        detail = reason if source is None else f"{source}: {reason}"
        super().__init__(detail)
        self.reason = reason
        self.source = source


class MissingSavedModelError(DemoBundleError):
    """Raised when the directory to package holds no save_pretrained() output."""

    def __init__(self, model_dir: Path, *, found: bool) -> None:
        """Report whether the directory is absent or merely missing config.json."""
        reason = (
            "no saved model here -- run Part 7 first"
            if not found
            else "no config.json here; not a save_pretrained() output"
        )
        super().__init__(reason, source=model_dir)
        self.model_dir = model_dir


@dataclass(frozen=True, slots=True)
class DemoBundleSpec:
    """Everything the demo needs besides the weights.

    Mirrors the `demo_config.json` that Part 7B of the notebook writes, so the
    notebook and this package cannot disagree about the shape of the artifact.
    """

    labels: tuple[str, ...]
    backbone: str
    max_length: int
    t_star: float
    hate_recall_target: float | None = None
    calibrated_on: str | None = None
    training: Mapping[str, JsonValue] = field(default_factory=dict)
    metrics: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DemoConfig:
    """Validated contents of demo_config.json.

    `load_demo_config` narrows the loose JSON into this once, at the boundary, so
    callers get real `int`/`float`/`str` values instead of re-checking a union at
    every single use.
    """

    labels: tuple[str, ...]
    hate_label_index: int
    backbone: str
    max_length: int
    t_star: float
    hate_recall_target: float | None = None
    calibrated_on: str | None = None
    training: Mapping[str, JsonValue] = field(default_factory=dict)
    metrics: Mapping[str, float] = field(default_factory=dict)
    has_sample: bool = False
    task: str = TASK_ID
    created_at: str = ""

    @property
    def hate_label(self) -> str:
        """Return the label whose probability the moderation threshold applies to."""
        return self.labels[self.hate_label_index]


def build_config(spec: DemoBundleSpec, *, has_sample: bool = False) -> dict[str, JsonValue]:
    """Return the demo_config.json payload for a spec. Pure -- no filesystem access."""
    hate_index = len(spec.labels) - 1
    threshold: dict[str, JsonValue] = {
        "t_star": spec.t_star,
        "hate_recall_target": spec.hate_recall_target,
        "calibrated_on": spec.calibrated_on,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "task": TASK_ID,
        "labels": list(spec.labels),
        "hate_label_index": hate_index,
        "backbone": spec.backbone,
        "max_length": spec.max_length,
        "threshold": threshold,
        "training": dict(spec.training),
        "metrics": dict(spec.metrics),
        "has_sample": has_sample,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def _table(title: str, rows: Mapping[str, JsonValue]) -> list[str]:
    """Render a two-column markdown table, or nothing at all when there is no data."""
    if not rows:
        return []
    lines = [f"## {title}", "", "| | |", "|---|---|"]
    lines.extend(f"| `{name}` | {value} |" for name, value in rows.items())
    lines.append("")
    return lines


def render_card(spec: DemoBundleSpec, repo_id: str | None = None) -> str:
    """Return the model-card markdown for a bundle. Pure -- takes the spec."""
    trained_on = f" on `{spec.calibrated_on}`" if spec.calibrated_on else ""
    repo = repo_id or DEFAULT_REPO_HINT
    label_text = "` / `".join(spec.labels)
    snapshot = f'snapshot_download("{repo}", allow_patterns=["model/*", "{CONFIG_NAME}"])'

    lines = [
        "---",
        "license: other",
        "library_name: transformers",
        "pipeline_tag: text-classification",
        f"base_model: {spec.backbone}",
        "tags:",
        "- vietnamese",
        "- hate-speech-detection",
        "- content-moderation",
        "- text-classification",
        "datasets:",
        "- uitnlp/vihsd",
        "---",
        "",
        "# ViHSD - Vietnamese Hate-Speech Detection (3-way)",
        "",
        (
            f"Fine-tuned `{spec.backbone}` classifying Vietnamese social-media comments as "
            f"`{label_text}` on [`uitnlp/vihsd`](https://huggingface.co/datasets/uitnlp/vihsd)."
        ),
        "",
        "## Content warning",
        "",
        "This model exists to *detect* hateful Vietnamese text. Its predictions quote and score",
        "real user-generated content, including slurs. Nothing here endorses the training",
        "labels; handle any output the way you would handle the dataset itself.",
        "",
        "## Decision threshold",
        "",
        "The `HATE` class is not decided by argmax alone. A comment is escalated for",
        f"moderation review when `P(HATE) >= t*`, with **`t*` = {spec.t_star:.3f}** calibrated",
        f"for a HATE recall target of {spec.hate_recall_target}{trained_on}.",
        "",
        "Recall-biased on purpose: for moderation a missed hateful comment costs more than an",
        "extra item in the review queue. The demo exposes `t*` as a slider so the trade-off is",
        "visible rather than baked in.",
        "",
    ]
    lines.extend(_table("Official test split (6,680 comments)", spec.metrics))
    lines.extend(_table("Training recipe", spec.training))
    lines.extend(
        [
            "## Run the demo",
            "",
            "From a checkout of [`vuongbinh/uit-labs`](https://github.com/vuongbinh/uit-labs):",
            "",
            "```bash",
            "uv sync --extra dev --extra demo",
            f"uv run vihate demo --model {repo}",
            "```",
            "",
            "Or load the weights directly:",
            "",
            "```python",
            "from huggingface_hub import snapshot_download",
            "from transformers import AutoModelForSequenceClassification, AutoTokenizer",
            "",
            f"bundle = {snapshot}",
            'tokenizer = AutoTokenizer.from_pretrained(f"{bundle}/model", use_fast=False)',
            'model = AutoModelForSequenceClassification.from_pretrained(f"{bundle}/model")',
            "```",
            "",
            f"`{CONFIG_NAME}` carries the label order, `max_length` and the calibrated `t*`.",
            "Read the threshold from there rather than hard-coding it, so a deployed demo",
            "cannot drift from the notebook that produced it.",
            "",
            "Trained by `notebooks/vihate_project_run_fnal.ipynb` (Part 7); published from",
            "Part 7B.",
            "",
        ]
    )
    return "\n".join(lines)


def write_demo_bundle(
    bundle_dir: Path,
    model_dir: Path,
    spec: DemoBundleSpec,
    *,
    sample_csv: Path | None = None,
    repo_id: str | None = None,
) -> Path:
    """Copy a saved model into a self-describing bundle directory and return it.

    `model_dir` is copied rather than moved so the training output directory keeps
    working for anything else that reads it, and so the bundle stays a clean unit
    that can be zipped or uploaded as-is. `repo_id` only labels the model card.
    """
    if not model_dir.is_dir():
        raise MissingSavedModelError(model_dir, found=False)
    if not (model_dir / "config.json").is_file():
        raise MissingSavedModelError(model_dir, found=True)

    has_sample = sample_csv is not None and sample_csv.is_file()
    config = build_config(spec, has_sample=has_sample)

    bundle_dir.mkdir(parents=True, exist_ok=True)
    target_model = bundle_dir / MODEL_DIRNAME
    if target_model.resolve() != model_dir.resolve():
        if target_model.exists():
            shutil.rmtree(target_model)
        shutil.copytree(model_dir, target_model)

    write_json(bundle_dir / CONFIG_NAME, config)
    (bundle_dir / CARD_NAME).write_text(render_card(spec, repo_id), encoding="utf-8")
    if has_sample and sample_csv is not None:
        shutil.copyfile(sample_csv, bundle_dir / SAMPLE_NAME)

    load_demo_config(bundle_dir)  # fail here, not in the demo, if the bundle is malformed
    return bundle_dir


def _require_keys(config: Mapping[str, JsonValue], source: Path) -> None:
    """Raise when a loaded config is missing keys the demo cannot work without."""
    missing = [key for key in REQUIRED_KEYS if key not in config]
    if missing:
        reason = f"missing required key(s): {', '.join(missing)}"
        raise DemoBundleError(reason, source=source)


def _read_json_object(path: Path, bundle_dir: Path) -> dict[str, JsonValue]:
    """Read a JSON object off disk, raising DemoBundleError with a usable message."""
    if not path.is_file():
        reason = (
            "no demo_config.json here, so this is not a demo bundle. Pass the directory "
            "Part 7B wrote, or a Hub repo id it was pushed to."
        )
        raise DemoBundleError(reason, source=bundle_dir)
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        invalid = f"not valid JSON ({exc})"
        raise DemoBundleError(invalid, source=path) from exc
    if not isinstance(loaded, dict):
        not_object = "top level is not a JSON object"
        raise DemoBundleError(not_object, source=path)
    return loaded


def _number(value: JsonValue, key: str, source: Path) -> float:
    """Coerce one JSON scalar to float, or raise naming the key that was wrong."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        reason = f"{key} must be a number, got {type(value).__name__}"
        raise DemoBundleError(reason, source=source)
    return float(value)


def _optional_number(value: JsonValue, key: str, source: Path) -> float | None:
    """Coerce an optional JSON scalar to float, treating null as absent."""
    return None if value is None else _number(value, key, source)


def _string(value: JsonValue, key: str, source: Path) -> str:
    """Require a JSON string, or raise naming the key that was wrong."""
    if not isinstance(value, str):
        reason = f"{key} must be a string, got {type(value).__name__}"
        raise DemoBundleError(reason, source=source)
    return value


def _optional_string(value: JsonValue, key: str, source: Path) -> str | None:
    """Require an optional JSON string, treating null as absent."""
    return None if value is None else _string(value, key, source)


def load_demo_config(bundle_dir: Path) -> DemoConfig:
    """Read and validate demo_config.json into a typed DemoConfig.

    Every field the demo depends on is type-checked here, so a hand-edited or
    truncated bundle fails at load time instead of mis-scoring text later.
    """
    path = bundle_dir / CONFIG_NAME
    raw = _read_json_object(path, bundle_dir)

    version = raw.get("schema_version")
    if version != SCHEMA_VERSION:
        reason = f"schema_version {version!r}; this code reads version {SCHEMA_VERSION}"
        raise DemoBundleError(reason, source=path)
    _require_keys(raw, path)

    labels = raw["labels"]
    if not isinstance(labels, list) or not labels:
        no_labels = "declares an empty or non-list label set"
        raise DemoBundleError(no_labels, source=path)

    threshold = raw["threshold"]
    if not isinstance(threshold, Mapping):
        no_threshold = "threshold must be a JSON object"
        raise DemoBundleError(no_threshold, source=path)
    if "t_star" not in threshold:
        no_t_star = "threshold block has no t_star"
        raise DemoBundleError(no_t_star, source=path)

    training = raw.get("training")
    metrics_raw = raw.get("metrics")
    return DemoConfig(
        labels=tuple(str(name) for name in labels),
        hate_label_index=int(
            _number(raw.get("hate_label_index", len(labels) - 1), "hate_label_index", path)
        ),
        backbone=_string(raw["backbone"], "backbone", path),
        max_length=int(_number(raw["max_length"], "max_length", path)),
        t_star=_number(threshold["t_star"], "threshold.t_star", path),
        hate_recall_target=_optional_number(
            threshold.get("hate_recall_target"), "threshold.hate_recall_target", path
        ),
        calibrated_on=_optional_string(
            threshold.get("calibrated_on"), "threshold.calibrated_on", path
        ),
        training=dict(training) if isinstance(training, Mapping) else {},
        metrics=(
            {str(key): _number(value, f"metrics.{key}", path) for key, value in metrics_raw.items()}
            if isinstance(metrics_raw, Mapping)
            else {}
        ),
        has_sample=bool(raw.get("has_sample", False)),
        task=_string(raw.get("task", TASK_ID), "task", path),
        created_at=_string(raw.get("created_at", ""), "created_at", path),
    )


def resolve_bundle(source: str | Path) -> tuple[Path, DemoConfig]:
    """Turn a local directory or a Hub repo id into (bundle_dir, config).

    A `source` that exists on disk is used as-is; anything else is treated as a Hub
    repo id and fetched with snapshot_download, which caches, so a second launch is
    offline-fast.
    """
    candidate = Path(source).expanduser()
    if candidate.is_dir():
        return candidate, load_demo_config(candidate)

    from huggingface_hub import snapshot_download

    patterns = [f"{MODEL_DIRNAME}/*", CONFIG_NAME, CARD_NAME]
    try:
        local = Path(snapshot_download(repo_id=str(source), allow_patterns=patterns))
    except Exception as exc:
        reason = f"neither a local demo bundle nor a Hub repo we could download ({exc})"
        raise DemoBundleError(reason, source=source) from exc
    return local, load_demo_config(local)


def zip_bundle(bundle_dir: Path, zip_path: Path) -> Path:
    """Zip a bundle for Drive or offline storage and return the archive path."""
    load_demo_config(bundle_dir)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(path for path in bundle_dir.rglob("*") if path.is_file())
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(bundle_dir.parent))
    return zip_path
