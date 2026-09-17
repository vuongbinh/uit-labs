"""The notebook and src/vihate are two front ends to one contract. Keep them honest.

Part 7B of the notebook writes the bundle; `vihate.demo_bundle` reads it. Part 8 of
the notebook scores text; `vihate.serve_demo` scores the same text. Both pairs are
executed here -- the real cell source, not a paraphrase -- against a tiny fixture
model, and asserted to agree. Without this, the notebook's inline writer could drift
from the package's reader and nobody would notice until a demo failed in front of an
audience.
"""

import ast
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from conftest import BACKBONE, BENIGN_TEXTS, LABELS, MAX_LENGTH, T_STAR

from vihate import demo_bundle
from vihate.serve_demo import HateSpeechPredictor

if TYPE_CHECKING:
    from collections.abc import Callable

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "vihate_project_run_fnal.ipynb"

MARKERS = {
    "config": "from pathlib import Path",
    "helpers": "import json\nfrom statistics import",
    "transformers": "import torch\nfrom datasets import Dataset",
    "part7b": "# --- Part 7B: persist the final model.",
    "part8": "# Part 8 runs in a cold session:",
}


def cells() -> list[dict[str, Any]]:
    """Return every cell of the notebook."""
    if not NOTEBOOK.is_file():
        pytest.skip(f"notebook not found at {NOTEBOOK}")
    loaded = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return list(loaded["cells"])


def cell_source(key: str) -> str:
    """Return the source of the one code cell that starts with the key's marker."""
    marker = MARKERS[key]
    matches = [
        "".join(cell["source"])
        for cell in cells()
        if cell["cell_type"] == "code" and "".join(cell["source"]).startswith(marker)
    ]
    assert len(matches) == 1, f"expected one cell starting {marker!r}, found {len(matches)}"
    return matches[0]


def notebook_function(key: str, name: str) -> str:
    """Pull one function definition out of a notebook cell, verbatim."""
    source = cell_source(key)
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            segment = ast.get_source_segment(source, node)
            assert segment is not None
            return segment
    message = f"{name}() not found in the {key} cell"
    raise AssertionError(message)


@pytest.fixture(scope="module")
def notebook_helpers() -> dict[str, Any]:
    """write_json and load_tokenizer, taken from the notebook rather than reimplemented.

    Lifting a single def out of a cell drops the imports its cell provided, so those
    globals are supplied here -- the function bodies themselves stay verbatim.
    """
    from transformers import AutoTokenizer

    namespace: dict[str, Any] = {"json": json, "AutoTokenizer": AutoTokenizer}
    exec(notebook_function("helpers", "write_json"), namespace)  # noqa: S102
    exec(notebook_function("transformers", "load_tokenizer"), namespace)  # noqa: S102
    return namespace


@pytest.fixture
def trained_session(
    tmp_path: Path, tiny_model_dir: Path, notebook_helpers: dict[str, "Callable[..., Any]"]
) -> dict[str, Any]:
    """The state Part 7B expects to find after Part 7 has run."""
    import torch
    from transformers import AutoModelForSequenceClassification

    final_dir = tmp_path / "final"
    (final_dir / "model").mkdir(parents=True)
    for item in tiny_model_dir.iterdir():
        (final_dir / "model" / item.name).write_bytes(item.read_bytes())

    return {
        "final_dir": final_dir,
        "OUT_ROOT": tmp_path,
        "BUNDLE_DIR": tmp_path / "demo_bundle",
        "DEMO_SCHEMA_VERSION": demo_bundle.SCHEMA_VERSION,
        "HF_MODEL_REPO": "",
        "HF_REPO_PRIVATE": False,
        "DEMO_SOURCE": "",
        "LABEL_NAMES": LABELS,
        "FINAL_BACKBONE": BACKBONE,
        "MAX_LENGTH": MAX_LENGTH,
        "T_STAR": T_STAR,
        "HATE_RECALL_TARGET": 0.85,
        "CALIB_RUN": "r2_visobert",
        "LEARNING_RATE": 3e-5,
        "FINAL_LEARNING_RATE": 2e-5,
        "BEST_LR": 2e-5,
        "EPOCHS": 6.0,
        "BATCH_SIZE": 64,
        "SEED": 42,
        "FOLDS": 5,
        "final_report": {
            "scalars": {"macro_f1": 0.6834, "hate_recall": 0.8612, "offensive_recall": 0.6201}
        },
        "final_report_calibrated": {"scalars": {"macro_f1": 0.6701}},
        "write_json": notebook_helpers["write_json"],
        "load_tokenizer": notebook_helpers["load_tokenizer"],
        "torch": torch,
        "AutoModelForSequenceClassification": AutoModelForSequenceClassification,
        "Path": Path,
    }


def run_part7b(session: dict[str, Any]) -> dict[str, Any]:
    """Execute the real Part 7B cell against a prepared session namespace."""
    exec(compile(cell_source("part7b"), "<part7b>", "exec"), session)  # noqa: S102
    return session


def run_part8(session: dict[str, Any]) -> dict[str, Any]:
    """Execute the real Part 8 setup cell against a prepared session namespace."""
    exec(compile(cell_source("part8"), "<part8>", "exec"), session)  # noqa: S102
    return session


def part8_namespace(bundle: Path, out_root: Path) -> dict[str, Any]:
    """Build the cold-start namespace Part 8 needs: no training state in scope."""
    # The real config cell does OUT_ROOT.mkdir(parents=True, exist_ok=True); Part 8
    # writes its batch CSV there and relies on that having happened.
    out_root.mkdir(parents=True, exist_ok=True)
    return {
        "Path": Path,
        "OUT_ROOT": out_root,
        "BUNDLE_DIR": bundle,
        "DEMO_SOURCE": "",
        "DEMO_SCHEMA_VERSION": demo_bundle.SCHEMA_VERSION,
    }


# ---------------------------------------------------------- Part 7B writes it


def test_part7b_writes_a_bundle_the_package_can_read(trained_session: dict[str, Any]) -> None:
    bundle = run_part7b(trained_session)["BUNDLE_DIR"]
    resolved, config = demo_bundle.resolve_bundle(bundle)

    assert resolved == Path(bundle)
    assert config.labels == LABELS
    assert config.backbone == BACKBONE
    assert config.max_length == MAX_LENGTH
    assert config.t_star == pytest.approx(T_STAR)
    assert config.hate_recall_target == pytest.approx(0.85)
    assert config.calibrated_on == "r2_visobert"


def test_part7b_records_the_training_recipe_and_test_metrics(
    trained_session: dict[str, Any],
) -> None:
    config = demo_bundle.load_demo_config(run_part7b(trained_session)["BUNDLE_DIR"])
    assert config.training["learning_rate"] == pytest.approx(2e-5)  # FINAL_LEARNING_RATE wins
    assert config.training["best_lr"] == pytest.approx(2e-5)
    assert config.training["seed"] == 42
    assert config.metrics["test_macro_f1"] == pytest.approx(0.6834)
    assert config.metrics["test_macro_f1_hate_calibrated"] == pytest.approx(0.6701)
    assert config.metrics["test_hate_recall"] == pytest.approx(0.8612)


def test_part7b_leaves_a_zip_and_a_card_next_to_the_bundle(
    trained_session: dict[str, Any],
) -> None:
    session = run_part7b(trained_session)
    bundle = Path(session["BUNDLE_DIR"])
    assert Path(session["BUNDLE_ZIP"]).is_file()
    card = (bundle / "README.md").read_text(encoding="utf-8")
    assert card.startswith("---\n")
    assert BACKBONE in card
    assert "0.6834" in card
    assert f"`t*` = {T_STAR:.3f}" in card


def test_part7b_skips_the_upload_when_no_repo_is_configured(
    trained_session: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    run_part7b(trained_session)
    assert "skipped" in capsys.readouterr().out


def test_part7b_self_check_proves_the_copy_reloads(
    trained_session: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    run_part7b(trained_session)
    out = capsys.readouterr().out
    assert "self-check OK" in out
    assert "probabilities sum to 1.0000" in out


def test_part7b_points_at_the_package_commands(
    trained_session: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    """The instructions Part 7B prints must be commands that actually exist."""
    run_part7b(trained_session)
    out = capsys.readouterr().out
    assert "uv run vihate demo --model" in out
    assert "uv run vihate publish-space" in out
    assert "vihsd/" not in out


def test_part7b_runs_before_part8() -> None:
    order = [
        "".join(cell["source"])[:40] for cell in cells() if cell["cell_type"] == "code"
    ]
    part7b = next(i for i, head in enumerate(order) if head.startswith(MARKERS["part7b"]))
    part8 = next(i for i, head in enumerate(order) if head.startswith(MARKERS["part8"]))
    assert part7b < part8, "Part 7B must run before the demo that consumes it"


# ------------------------------------------------------- Part 8 reads it back


def test_part8_cold_start_loads_the_bundle_without_training(
    trained_session: dict[str, Any], tmp_path: Path
) -> None:
    bundle = run_part7b(trained_session)["BUNDLE_DIR"]
    namespace = run_part8(part8_namespace(bundle, tmp_path / "cold"))

    assert "final_trainer" not in namespace  # took the cold branch
    assert namespace["DEMO_T_STAR"] == pytest.approx(T_STAR)
    assert namespace["DEMO_MAX_LENGTH"] == MAX_LENGTH
    assert namespace["DEMO_LABEL_NAMES"] == list(LABELS)
    assert namespace["LABELS"] == {0: "CLEAN", 1: "OFFENSIVE", 2: "HATE"}
    assert namespace["HATE_LABEL"] == "HATE"
    assert "persisted bundle" in namespace["DEMO_ORIGIN"]


def test_part8_cold_start_needs_no_sample_file(
    trained_session: dict[str, Any], tmp_path: Path
) -> None:
    bundle = run_part7b(trained_session)["BUNDLE_DIR"]
    namespace = run_part8(part8_namespace(bundle, tmp_path / "cold"))
    assert "demo_test_path" not in namespace
    assert ".txt" in namespace["SAMPLE_NOTE"]  # falls back to upload instructions


def test_part8_live_path_still_uses_the_in_memory_model(
    trained_session: dict[str, Any], tmp_path: Path, tiny_model_dir: Path
) -> None:
    """A full run must not pay for a reload it does not need."""
    import torch
    from transformers import AutoModelForSequenceClassification

    bundle = run_part7b(trained_session)["BUNDLE_DIR"]
    namespace = part8_namespace(bundle, tmp_path / "live")
    model = AutoModelForSequenceClassification.from_pretrained(tiny_model_dir)
    namespace["final_trainer"] = type("Trainer", (), {"model": model})()
    namespace["final_tokenizer"] = trained_session["load_tokenizer"](str(tiny_model_dir))
    namespace["LABEL_NAMES"] = LABELS
    namespace["MAX_LENGTH"] = MAX_LENGTH
    namespace["T_STAR"] = T_STAR
    run_part8(namespace)

    assert "live training state" in namespace["DEMO_ORIGIN"]
    assert namespace["DEMO_T_STAR"] == pytest.approx(T_STAR)
    assert isinstance(namespace["demo_model"], torch.nn.Module)


def test_notebook_part8_and_serve_demo_agree_exactly(
    trained_session: dict[str, Any], tmp_path: Path
) -> None:
    """The whole point: same bundle in, same predictions out, from either front end."""
    bundle = run_part7b(trained_session)["BUNDLE_DIR"]
    namespace = run_part8(part8_namespace(bundle, tmp_path / "cold"))
    server = HateSpeechPredictor(bundle, device="cpu", output_dir=tmp_path / "server-out")

    for text in BENIGN_TEXTS:
        note_prediction, note_scores = namespace["predict_detailed"](text)
        serve_prediction, serve_scores = server.predict_detailed(text)
        assert note_prediction == serve_prediction
        assert list(note_scores) == list(serve_scores) == list(LABELS)
        for label in LABELS:
            assert note_scores[label] == pytest.approx(serve_scores[label], abs=1e-6)


def test_notebook_part8_agrees_with_server_at_odd_thresholds(
    trained_session: dict[str, Any], tmp_path: Path
) -> None:
    bundle = run_part7b(trained_session)["BUNDLE_DIR"]
    namespace = run_part8(part8_namespace(bundle, tmp_path / "cold"))
    server = HateSpeechPredictor(bundle, device="cpu", output_dir=tmp_path / "server-out")

    for text in BENIGN_TEXTS:
        for threshold in (0.0, 0.37, 1.0):
            note_prediction = namespace["predict_detailed"](text, threshold)[0]
            assert note_prediction == server.predict_detailed(text, threshold)[0]


def test_notebook_part8_batch_tab_matches_the_server(
    trained_session: dict[str, Any], tmp_path: Path
) -> None:
    bundle = run_part7b(trained_session)["BUNDLE_DIR"]
    namespace = run_part8(part8_namespace(bundle, tmp_path / "cold"))
    server = HateSpeechPredictor(bundle, device="cpu", output_dir=tmp_path / "server-out")

    sample = tmp_path / "sample.txt"
    sample.write_text("\n".join(BENIGN_TEXTS), encoding="utf-8")

    note_table, note_csv = namespace["predict_batch"](sample)
    serve_table, serve_csv = server.predict_batch(sample)

    assert list(note_table.columns) == list(serve_table.columns)
    assert note_table.drop(columns=["flagged_for_review"]).equals(
        serve_table.drop(columns=["flagged_for_review"])
    )
    assert list(note_table["flagged_for_review"]) == list(serve_table["flagged_for_review"])
    assert note_csv is not None
    assert serve_csv is not None
    assert Path(note_csv).is_file()
    assert Path(serve_csv).is_file()


def test_part8_rejects_a_bundle_from_a_different_schema(
    trained_session: dict[str, Any], tmp_path: Path
) -> None:
    bundle = Path(run_part7b(trained_session)["BUNDLE_DIR"])
    config_path = bundle / "demo_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["schema_version"] = demo_bundle.SCHEMA_VERSION + 1
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(AssertionError, match="schema_version"):
        run_part8(part8_namespace(bundle, tmp_path / "cold"))


# ------------------------------------------------------------------ config cell


@pytest.fixture
def config_namespace(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Run the config cell without letting it create /content/outputs on this machine."""
    created: list[Path] = []

    def record_mkdir(self: Path, *_args: object, **_kwargs: object) -> None:
        """Record the mkdir the config cell would do, without touching the disk."""
        created.append(self)

    monkeypatch.setattr(Path, "mkdir", record_mkdir)
    namespace: dict[str, Any] = {}
    exec(compile(cell_source("config"), "<config>", "exec"), namespace)  # noqa: S102
    namespace["_created_dirs"] = created
    return namespace


def test_config_cell_defines_the_persistence_settings(config_namespace: dict[str, Any]) -> None:
    for name in (
        "DEMO_SCHEMA_VERSION", "BUNDLE_DIR", "HF_MODEL_REPO", "HF_REPO_PRIVATE", "DEMO_SOURCE"
    ):
        assert name in config_namespace, f"config cell no longer defines {name}"
    assert config_namespace["DEMO_SCHEMA_VERSION"] == demo_bundle.SCHEMA_VERSION
    assert config_namespace["BUNDLE_DIR"].name == "demo_bundle"
    assert config_namespace["HF_MODEL_REPO"] == ""  # publishing is opt-in
    assert config_namespace["DEMO_SOURCE"] == ""  # default: the local bundle


def test_config_cell_puts_the_bundle_under_out_root(config_namespace: dict[str, Any]) -> None:
    assert config_namespace["BUNDLE_DIR"].parent == config_namespace["OUT_ROOT"]
    assert config_namespace["_created_dirs"] == [config_namespace["OUT_ROOT"]]


def test_config_cell_schema_version_matches_the_package(
    config_namespace: dict[str, Any],
) -> None:
    """A notebook that reads a schema version the package no longer writes is a silent break."""
    assert config_namespace["DEMO_SCHEMA_VERSION"] == demo_bundle.SCHEMA_VERSION


def test_fixture_model_is_shaped_like_the_real_thing(tiny_model_dir: Path) -> None:
    config = json.loads((tiny_model_dir / "config.json").read_text(encoding="utf-8"))
    assert config["architectures"] == ["XLMRobertaForSequenceClassification"]
    assert config["id2label"] == {"0": "CLEAN", "1": "OFFENSIVE", "2": "HATE"}
    assert config["label2id"] == {"CLEAN": 0, "OFFENSIVE": 1, "HATE": 2}
