"""The artifact contract: what Part 7B writes is what the demo can read back."""

import json
import zipfile
from pathlib import Path

import pytest
from conftest import BACKBONE, LABELS, MAX_LENGTH, REPO_ID, SPEC, T_STAR, make_tiny_model

from vihate.demo_bundle import (
    CARD_NAME,
    CONFIG_NAME,
    MODEL_DIRNAME,
    SAMPLE_NAME,
    SCHEMA_VERSION,
    DemoBundleError,
    MissingSavedModelError,
    build_config,
    load_demo_config,
    render_card,
    resolve_bundle,
    write_demo_bundle,
    zip_bundle,
)
from vihate.reporting import write_json


def write_raw_config(directory: Path, payload: dict[str, object]) -> Path:
    """Write a demo_config.json by hand so a malformed bundle can be tested."""
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / CONFIG_NAME, payload)  # type: ignore[arg-type]
    return directory


def broken(**overrides: object) -> dict[str, object]:
    """Return a valid config payload with individual keys replaced."""
    payload: dict[str, object] = dict(build_config(SPEC))
    payload.update(overrides)
    return payload


# ------------------------------------------------------------------ bundle shape


def test_bundle_layout(bundle_dir: Path) -> None:
    assert (bundle_dir / MODEL_DIRNAME / "config.json").is_file()
    assert (bundle_dir / CONFIG_NAME).is_file()
    assert (bundle_dir / CARD_NAME).is_file()
    assert (bundle_dir / SAMPLE_NAME).is_file()


def test_config_records_everything_the_demo_needs(bundle_dir: Path) -> None:
    config = load_demo_config(bundle_dir)
    assert config.labels == LABELS
    assert config.hate_label == "HATE"
    assert config.backbone == BACKBONE
    assert config.max_length == MAX_LENGTH
    assert config.t_star == pytest.approx(T_STAR)
    assert config.hate_recall_target == pytest.approx(0.85)
    assert config.calibrated_on == "r2_visobert"
    assert config.metrics["test_macro_f1"] == pytest.approx(0.6834)
    assert config.training["seed"] == 42
    assert config.has_sample is True
    assert config.created_at


def test_config_on_disk_is_plain_json(bundle_dir: Path) -> None:
    """The typed DemoConfig is a read-side view; the file stays portable JSON."""
    raw = json.loads((bundle_dir / CONFIG_NAME).read_text(encoding="utf-8"))
    assert raw["schema_version"] == SCHEMA_VERSION
    assert raw["task"] == "vihsd-3way"
    assert raw["hate_label_index"] == 2
    assert raw["threshold"]["t_star"] == pytest.approx(T_STAR)


def test_source_model_dir_is_left_intact(bundle_dir: Path, tiny_model_dir: Path) -> None:
    # Copying, not moving: Part 7's output must still be readable after publishing.
    assert (tiny_model_dir / "config.json").is_file()
    assert (bundle_dir / MODEL_DIRNAME / "config.json").read_bytes() == (
        tiny_model_dir / "config.json"
    ).read_bytes()


def test_card_states_threshold_and_metrics(bundle_dir: Path) -> None:
    card = (bundle_dir / CARD_NAME).read_text(encoding="utf-8")
    assert card.startswith("---\n")
    assert f"`t*` = {T_STAR:.3f}" in card
    assert "0.6834" in card
    assert BACKBONE in card
    assert "uitnlp/vihsd" in card
    assert REPO_ID in card  # repo_id threaded into the run instructions
    assert "Content warning" in card


def test_rewriting_a_bundle_replaces_the_old_model(tiny_model_dir: Path, tmp_path: Path) -> None:
    target = tmp_path / "bundle"
    write_demo_bundle(target, tiny_model_dir, SPEC)
    stale = target / MODEL_DIRNAME / "stale.txt"
    stale.write_text("leftover from an earlier run", encoding="utf-8")
    write_demo_bundle(target, tiny_model_dir, SPEC)
    assert not stale.exists()


def test_missing_model_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(MissingSavedModelError, match="run Part 7 first"):
        write_demo_bundle(tmp_path / "b", tmp_path / "nope", SPEC)
    assert not (tmp_path / "b").exists()  # fails before leaving a half-written bundle


def test_directory_without_config_json_is_not_a_model(tmp_path: Path) -> None:
    not_a_model = tmp_path / "weights-ish"
    not_a_model.mkdir()
    (not_a_model / "pytorch_model.bin").write_bytes(b"not really weights")
    with pytest.raises(MissingSavedModelError, match="save_pretrained"):
        write_demo_bundle(tmp_path / "b", not_a_model, SPEC)


def test_no_sample_means_no_sample_file(tiny_model_dir: Path, tmp_path: Path) -> None:
    bundle = write_demo_bundle(tmp_path / "b", tiny_model_dir, SPEC)
    assert not (bundle / SAMPLE_NAME).exists()
    assert load_demo_config(bundle).has_sample is False


def test_missing_sample_path_is_not_an_error(tiny_model_dir: Path, tmp_path: Path) -> None:
    bundle = write_demo_bundle(
        tmp_path / "b", tiny_model_dir, SPEC, sample_csv=tmp_path / "never-written.csv"
    )
    assert load_demo_config(bundle).has_sample is False


# ------------------------------------------------------------ config validation


def test_rejects_wrong_schema_version(tmp_path: Path) -> None:
    bundle = write_raw_config(tmp_path / "b", broken(schema_version=SCHEMA_VERSION + 1))
    with pytest.raises(DemoBundleError, match="schema_version"):
        load_demo_config(bundle)


def test_rejects_missing_required_keys(tmp_path: Path) -> None:
    payload = broken()
    del payload["backbone"]
    del payload["max_length"]
    with pytest.raises(DemoBundleError, match="backbone, max_length"):
        load_demo_config(write_raw_config(tmp_path / "b", payload))


def test_rejects_threshold_without_t_star(tmp_path: Path) -> None:
    with pytest.raises(DemoBundleError, match="no t_star"):
        load_demo_config(write_raw_config(tmp_path / "b", broken(threshold={"target": 0.85})))


def test_rejects_a_threshold_that_is_not_an_object(tmp_path: Path) -> None:
    with pytest.raises(DemoBundleError, match="threshold must be a JSON object"):
        load_demo_config(write_raw_config(tmp_path / "b", broken(threshold=0.19)))


def test_rejects_empty_label_list(tmp_path: Path) -> None:
    with pytest.raises(DemoBundleError, match="empty or non-list label set"):
        load_demo_config(write_raw_config(tmp_path / "b", broken(labels=[])))


def test_rejects_a_max_length_that_is_not_a_number(tmp_path: Path) -> None:
    with pytest.raises(DemoBundleError, match="max_length must be a number"):
        load_demo_config(write_raw_config(tmp_path / "b", broken(max_length="128")))


def test_rejects_a_backbone_that_is_not_a_string(tmp_path: Path) -> None:
    with pytest.raises(DemoBundleError, match="backbone must be a string"):
        load_demo_config(write_raw_config(tmp_path / "b", broken(backbone=7)))


def test_rejects_invalid_json(tmp_path: Path) -> None:
    bundle = tmp_path / "b"
    bundle.mkdir()
    (bundle / CONFIG_NAME).write_text("{not json", encoding="utf-8")
    with pytest.raises(DemoBundleError, match="not valid JSON"):
        load_demo_config(bundle)


def test_rejects_a_json_array_instead_of_an_object(tmp_path: Path) -> None:
    bundle = tmp_path / "b"
    bundle.mkdir()
    (bundle / CONFIG_NAME).write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(DemoBundleError, match="not a JSON object"):
        load_demo_config(bundle)


def test_missing_config_message_points_at_the_fix(tmp_path: Path) -> None:
    with pytest.raises(DemoBundleError, match="Part 7B"):
        load_demo_config(tmp_path / "empty")


def test_absent_optional_fields_load_as_none(tmp_path: Path) -> None:
    payload = broken(threshold={"t_star": T_STAR}, training={}, metrics={})
    config = load_demo_config(write_raw_config(tmp_path / "b", payload))
    assert config.hate_recall_target is None
    assert config.calibrated_on is None
    assert config.metrics == {}
    assert config.training == {}


# ------------------------------------------------------------------- resolution


def test_resolve_bundle_accepts_a_local_directory(bundle_dir: Path) -> None:
    resolved, config = resolve_bundle(bundle_dir)
    assert resolved == bundle_dir
    assert config.t_star == pytest.approx(T_STAR)


def test_resolve_bundle_accepts_a_string_path(bundle_dir: Path) -> None:
    resolved, config = resolve_bundle(str(bundle_dir))
    assert resolved == Path(bundle_dir)
    assert config.labels == LABELS


def test_resolve_bundle_rejects_an_unusable_source() -> None:
    with pytest.raises(DemoBundleError, match="neither a local demo bundle nor a Hub repo"):
        resolve_bundle("this-org-does-not-exist-9x7/no-such-model-9x7")


# -------------------------------------------------------------------------- zip


def test_zip_roundtrip(bundle_dir: Path, tmp_path: Path) -> None:
    archive = zip_bundle(bundle_dir, tmp_path / "nested" / "bundle.zip")
    assert archive.is_file()
    with zipfile.ZipFile(archive) as handle:
        names = handle.namelist()
        assert f"{bundle_dir.name}/{CONFIG_NAME}" in names
        assert f"{bundle_dir.name}/{MODEL_DIRNAME}/config.json" in names
        handle.extractall(tmp_path / "unzipped")
    restored = tmp_path / "unzipped" / bundle_dir.name
    assert load_demo_config(restored).t_star == pytest.approx(T_STAR)


def test_zip_refuses_an_invalid_bundle(tmp_path: Path) -> None:
    with pytest.raises(DemoBundleError):
        zip_bundle(tmp_path / "nothing-here", tmp_path / "out.zip")


# -------------------------------------------------------- pure helper functions


def test_render_card_is_pure_and_uses_a_placeholder_repo() -> None:
    card = render_card(SPEC)
    assert "<your-hf-user>/vihsd-visobert" in card
    assert "## Official test split" in card


def test_render_card_omits_empty_tables() -> None:
    from conftest import variant

    card = render_card(variant(metrics={}, training={}))
    assert "## Official test split" not in card
    assert "## Training recipe" not in card


def test_render_card_omits_the_calibration_run_when_absent() -> None:
    from conftest import variant

    card = render_card(variant(calibrated_on=None))
    assert "on `r2_visobert`" not in card


def test_write_bundle_validates_its_own_output(tiny_model_dir: Path, tmp_path: Path) -> None:
    # write_demo_bundle calls load_demo_config at the end; a bundle that cannot be
    # read back must never be reported as written.
    bundle = write_demo_bundle(tmp_path / "b", tiny_model_dir, SPEC)
    assert load_demo_config(bundle).backbone == BACKBONE


def test_fixture_model_is_reusable_across_bundles(tmp_path: Path) -> None:
    model_dir = make_tiny_model(tmp_path / "another-model")
    assert (model_dir / "config.json").is_file()
    bundle = write_demo_bundle(tmp_path / "b", model_dir, SPEC)
    assert load_demo_config(bundle).labels == LABELS
