"""The demo server: threshold semantics, batch parity, CLI wiring, and a real HTTP boot."""

import shutil
import socket
import urllib.request
from pathlib import Path
from typing import cast

import pandas as pd
import pytest
from conftest import BACKBONE, BENIGN_TEXTS, LABELS, MAX_LENGTH, T_STAR, make_tiny_model
from typer.testing import CliRunner

from vihate.demo_bundle import DemoBundleError, write_demo_bundle
from vihate.serve_demo import (
    FLAG_NOTE,
    HateSpeechPredictor,
    ServeConfig,
    build_app,
    launch_app,
    load_tokenizer,
    verify_bundle,
)

runner = CliRunner()


# --------------------------------------------------------------------- loading


def test_predictor_takes_its_settings_from_the_bundle(predictor: HateSpeechPredictor) -> None:
    assert predictor.labels == LABELS
    assert predictor.hate_label == "HATE"
    assert predictor.max_length == MAX_LENGTH
    assert predictor.t_star == pytest.approx(T_STAR)
    assert predictor.config.backbone == BACKBONE
    assert str(predictor.device) == "cpu"
    assert not predictor.model.training


def test_sample_file_is_surfaced_when_the_bundle_has_one(predictor: HateSpeechPredictor) -> None:
    assert predictor.sample_path is not None
    assert predictor.sample_path.is_file()


def test_columns_are_derived_from_the_bundle_labels(predictor: HateSpeechPredictor) -> None:
    assert predictor.prob_columns == tuple(f"P({label})" for label in LABELS)
    assert predictor.batch_columns == (
        "text", "predicted_class", "P(CLEAN)", "P(OFFENSIVE)", "P(HATE)", "flagged_for_review",
    )


def test_load_tokenizer_returns_a_working_tokenizer(tiny_model_dir: Path) -> None:
    encoded = load_tokenizer(tiny_model_dir)("xin chào bạn")
    input_ids = cast("list[int]", encoded["input_ids"])
    assert len(input_ids) > 0


def test_unusable_source_raises_a_readable_error(tmp_path: Path) -> None:
    with pytest.raises(DemoBundleError, match="neither a local demo bundle nor a Hub repo"):
        HateSpeechPredictor(tmp_path / "nothing-here")


# ------------------------------------------------------------- single comment


def test_probabilities_cover_every_label_and_sum_to_one(predictor: HateSpeechPredictor) -> None:
    scores = predictor.probabilities(BENIGN_TEXTS[0])
    assert list(scores) == list(LABELS)
    assert sum(scores.values()) == pytest.approx(1.0, abs=1e-5)


def test_predict_detailed_uses_the_bundle_threshold_by_default(
    predictor: HateSpeechPredictor,
) -> None:
    default, _ = predictor.predict_detailed(BENIGN_TEXTS[0])
    explicit, _ = predictor.predict_detailed(BENIGN_TEXTS[0], T_STAR)
    assert default == explicit


def test_threshold_of_one_never_flags(predictor: HateSpeechPredictor) -> None:
    for text in BENIGN_TEXTS:
        prediction, _ = predictor.predict_detailed(text, 1.01)
        assert FLAG_NOTE not in prediction


def test_threshold_of_zero_always_flags(predictor: HateSpeechPredictor) -> None:
    for text in BENIGN_TEXTS:
        prediction, _ = predictor.predict_detailed(text, 0.0)
        assert FLAG_NOTE in prediction


def test_flag_does_not_change_predicted_class(predictor: HateSpeechPredictor) -> None:
    """The moderation note is appended; the class stays the argmax.

    This is the notebook's existing behaviour and deliberately NOT Part 6's
    apply_hate_rule(), which overwrites the prediction with HATE. Pinned here so
    changing it is a decision rather than an accident.
    """
    text = BENIGN_TEXTS[0]
    unflagged, scores = predictor.predict_detailed(text, 1.01)
    flagged, _ = predictor.predict_detailed(text, 0.0)
    argmax_label = max(scores, key=lambda label: scores[label])

    assert unflagged == argmax_label
    assert flagged == f"{argmax_label} ({FLAG_NOTE}: P(HATE) >= 0.000)"
    assert flagged.split(" (", maxsplit=1)[0] == argmax_label


def test_flag_note_shows_the_threshold_actually_used(predictor: HateSpeechPredictor) -> None:
    text = BENIGN_TEXTS[0]
    # Setting t exactly at P(HATE) always trips the >= rule, whatever the weights say.
    hate_prob = predictor.probabilities(text)[predictor.hate_label]
    flagged, _ = predictor.predict_detailed(text, hate_prob)
    assert FLAG_NOTE in flagged
    assert f"{hate_prob:.3f}" in flagged


def test_blank_input_is_answered_not_crashed(predictor: HateSpeechPredictor) -> None:
    for blank in ("", "   ", None):
        prediction, scores = predictor.predict_detailed(blank)
        assert prediction == "Please enter some text."
        assert scores == {}


# ------------------------------------------------------------------ batch tab


def write_text_file(directory: Path, name: str, body: str) -> Path:
    """Write an upload fixture and return its path."""
    path = directory / name
    path.write_text(body, encoding="utf-8")
    return path


def test_batch_reads_txt_one_comment_per_line(
    predictor: HateSpeechPredictor, tmp_path: Path
) -> None:
    path = write_text_file(tmp_path, "in.txt", "\n".join(BENIGN_TEXTS) + "\n\n   \n")
    table, csv_path = predictor.predict_batch(path)
    assert len(table) == len(BENIGN_TEXTS)  # blank lines dropped
    assert list(table["text"]) == list(BENIGN_TEXTS)
    assert csv_path is not None
    assert Path(csv_path).is_file()


@pytest.mark.parametrize(("name", "sep"), [("in.csv", ","), ("in.tsv", "\t")])
def test_batch_reads_csv_and_tsv(
    predictor: HateSpeechPredictor, tmp_path: Path, name: str, sep: str
) -> None:
    body = f"free_text{sep}true_label\n" + "".join(
        f"{text}{sep}CLEAN\n" for text in BENIGN_TEXTS
    )
    table, _ = predictor.predict_batch(write_text_file(tmp_path, name, body))
    assert list(table["text"]) == list(BENIGN_TEXTS)


def test_batch_picks_the_first_recognised_text_column(
    predictor: HateSpeechPredictor, tmp_path: Path
) -> None:
    body = "id,comment\n" + "".join(f"{i},{text}\n" for i, text in enumerate(BENIGN_TEXTS))
    table, _ = predictor.predict_batch(write_text_file(tmp_path, "in.csv", body))
    assert list(table["text"]) == list(BENIGN_TEXTS)


def test_batch_accepts_a_path_object_and_a_string_alike(
    predictor: HateSpeechPredictor, tmp_path: Path
) -> None:
    path = write_text_file(tmp_path, "in.txt", "\n".join(BENIGN_TEXTS))
    from_path, _ = predictor.predict_batch(path)
    from_string, _ = predictor.predict_batch(str(path))
    pd.testing.assert_frame_equal(from_path, from_string)


def test_batch_matches_the_single_comment_path_row_by_row(
    predictor: HateSpeechPredictor, tmp_path: Path
) -> None:
    """The batch tab must not be a second, subtly different implementation."""
    path = write_text_file(tmp_path, "in.txt", "\n".join(BENIGN_TEXTS))
    table, csv_path = predictor.predict_batch(path, 0.31)

    for _, row in table.iterrows():
        prediction, scores = predictor.predict_detailed(str(row["text"]), 0.31)
        assert row["predicted_class"] == prediction.split(" (", maxsplit=1)[0]
        assert row["flagged_for_review"] == (FLAG_NOTE in prediction)
        for label in LABELS:
            assert row[f"P({label})"] == pytest.approx(round(scores[label], 4))

    assert csv_path is not None
    exported = pd.read_csv(csv_path, encoding="utf-8-sig")
    assert list(exported.columns) == list(predictor.batch_columns)
    assert len(exported) == len(BENIGN_TEXTS)


def test_batch_without_a_file_returns_an_empty_table(predictor: HateSpeechPredictor) -> None:
    table, csv_path = predictor.predict_batch(None)
    assert csv_path is None
    assert table.empty
    assert list(table.columns) == list(predictor.batch_columns)


def test_batch_export_lands_in_the_configured_output_dir(
    bundle_dir: Path, tmp_path: Path
) -> None:
    out_dir = tmp_path / "exports"
    local = HateSpeechPredictor(bundle_dir, device="cpu", output_dir=out_dir)
    source = write_text_file(tmp_path, "in.txt", BENIGN_TEXTS[0])
    _, csv_path = local.predict_batch(source)
    assert csv_path is not None
    assert Path(csv_path).parent == out_dir


# --------------------------------------------------------------- gradio app


def test_build_app_has_both_tabs_and_the_calibrated_slider(
    predictor: HateSpeechPredictor,
) -> None:
    import gradio as gr

    app = build_app(predictor)
    assert isinstance(app, gr.Blocks)
    assert app.title == "Vietnamese Hate Speech Detection"
    assert sum(isinstance(block, gr.Tab) for block in app.blocks.values()) == 2

    sliders = [block for block in app.blocks.values() if isinstance(block, gr.Slider)]
    assert len(sliders) == 2
    assert all(slider.value == pytest.approx(T_STAR) for slider in sliders)


def free_port() -> int:
    """Return a port the OS has just handed out and nothing is bound to yet."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        _, port = probe.getsockname()
    return int(port)


def test_gradio_server_boots_and_answers_http(predictor: HateSpeechPredictor) -> None:
    """Not just 'the Blocks object built' -- the real server must serve the app."""
    port = free_port()
    app = build_app(predictor)
    app.launch(
        server_name="127.0.0.1",
        server_port=port,
        prevent_thread_lock=True,
        quiet=True,
        share=False,
    )
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=30) as response:
            body = response.read().decode("utf-8")
            assert response.status == 200
        assert "Vietnamese Hate Speech Detection" in body
    finally:
        app.close()


def test_launch_app_grants_gradio_access_to_the_export_dir(
    predictor: HateSpeechPredictor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gradio only serves files under the cwd or the temp dir.

    Without allowed_paths an explicit output_dir makes the batch tab's CSV download
    fail at click time -- found by driving the real server over HTTP.
    """
    launched: dict[str, object] = {}

    class FakeApp:
        def launch(self, **kwargs: object) -> None:
            launched.update(kwargs)

    from vihate import serve_demo

    monkeypatch.setattr(serve_demo, "build_app", lambda _predictor: FakeApp())
    launch_app(predictor, ServeConfig(source=str(predictor.bundle_dir)))
    assert launched["allowed_paths"] == [str(predictor.output_dir)]
    assert launched["show_error"] is True


# ------------------------------------------------------------------------ CLI


def test_demo_command_loads_then_launches(
    bundle_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vihate.cli import app

    launched: dict[str, object] = {}

    from vihate import serve_demo

    def fake_launch(loaded: HateSpeechPredictor, config: ServeConfig) -> None:
        launched["predictor"] = loaded
        launched["config"] = config

    monkeypatch.setattr(serve_demo, "launch_app", fake_launch)
    result = runner.invoke(
        app,
        ["demo", "--model", str(bundle_dir), "--host", "127.0.0.2", "--port", "7999",
         "--device", "cpu"],
    )
    assert result.exit_code == 0, result.output
    assert "ready" in result.output
    config = launched["config"]
    assert isinstance(config, ServeConfig)
    assert config.host == "127.0.0.2"
    assert config.port == 7999
    assert config.share is False


def test_demo_command_requires_a_model() -> None:
    from vihate.cli import app

    result = runner.invoke(app, ["demo"])
    assert result.exit_code != 0


def test_demo_command_accepts_share_and_out_dir(
    bundle_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vihate.cli import app

    launched: dict[str, object] = {}

    from vihate import serve_demo

    monkeypatch.setattr(
        serve_demo, "launch_app", lambda _p, config: launched.update({"config": config})
    )
    result = runner.invoke(
        app,
        ["demo", "--model", str(bundle_dir), "--share", "--out-dir", str(tmp_path),
         "--device", "cpu"],
    )
    assert result.exit_code == 0, result.output
    config = launched["config"]
    assert isinstance(config, ServeConfig)
    assert config.share is True
    assert config.output_dir == tmp_path


# --------------------------------------------- bundle integrity caught at load time


def two_label_bundle(tmp_path: Path) -> Path:
    """A bundle whose model head has 2 labels while demo_config.json declares 3."""
    from conftest import SPEC
    from transformers import XLMRobertaConfig, XLMRobertaForSequenceClassification

    model_dir = tmp_path / "two-label" / "model"
    model_dir.mkdir(parents=True)
    XLMRobertaForSequenceClassification(
        XLMRobertaConfig(
            vocab_size=20,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=64,
            max_position_embeddings=140,
            pad_token_id=0,
            num_labels=2,
        )
    ).save_pretrained(model_dir)
    with_tokenizer = make_tiny_model(tmp_path / "with-tokenizer")
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        shutil.copyfile(with_tokenizer / name, model_dir / name)
    return write_demo_bundle(tmp_path / "bundle", model_dir, SPEC)


def test_head_label_mismatch_is_caught_at_load_not_mid_predict(tmp_path: Path) -> None:
    """This used to load fine and then die at predict time with a bare KeyError."""
    bundle = two_label_bundle(tmp_path)
    with pytest.raises(DemoBundleError, match="model head has 2 labels"):
        HateSpeechPredictor(bundle, device="cpu")


def test_verify_bundle_passes_on_a_good_bundle(bundle_dir: Path) -> None:
    config = verify_bundle(bundle_dir, device="cpu")
    assert config.labels == LABELS


def test_verify_bundle_rejects_a_mismatched_head(tmp_path: Path) -> None:
    with pytest.raises(DemoBundleError, match="model head has 2 labels"):
        verify_bundle(two_label_bundle(tmp_path), device="cpu")
