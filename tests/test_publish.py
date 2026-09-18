"""Publishing: what gets uploaded, where, and what a Space is built from.

The Hub calls are stubbed -- these tests assert the arguments we would send, never
touch the network, and never need a write token.
"""

import ast
from pathlib import Path
from typing import cast

import pytest
from conftest import BACKBONE, LABELS
from typer.testing import CliRunner

from vihate import publish
from vihate.demo_bundle import DemoBundleError
from vihate.publish import SpaceSpec

runner = CliRunner()


class FakeApi:
    """Records the calls publish.py makes instead of making them."""

    def __init__(self) -> None:
        self.created: list[tuple[str, dict[str, object]]] = []
        self.uploaded: list[dict[str, object]] = []
        self.tokens: list[str | None] = []

    def create_repo(self, repo_id: str, **kwargs: object) -> str:
        self.created.append((repo_id, kwargs))
        return repo_id

    def upload_folder(self, folder_path: str, repo_id: str, **kwargs: object) -> str:
        self.uploaded.append({"folder_path": folder_path, "repo_id": repo_id, **kwargs})
        return f"https://huggingface.co/{repo_id}"


@pytest.fixture
def fake_api(monkeypatch: pytest.MonkeyPatch) -> FakeApi:
    api = FakeApi()

    def factory(token: str | None = None) -> FakeApi:
        api.tokens.append(token)
        return api

    monkeypatch.setattr(publish, "_api", factory)
    return api


def stub_package_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point publish at a fake package directory holding the modules a Space needs."""
    package = tmp_path / "pkg"
    package.mkdir()
    for name in publish.SPACE_MODULES:
        (package / name).write_text('"""Stub."""\n', encoding="utf-8")
    monkeypatch.setattr(publish, "PACKAGE_DIR", package)
    return package


# ------------------------------------------------------------------ model push


def test_push_creates_the_repo_and_uploads_the_bundle(
    bundle_dir: Path, fake_api: FakeApi
) -> None:
    url = publish.push_bundle_to_hub(bundle_dir, "someone/vihsd-visobert", token="t")
    assert url == "https://huggingface.co/someone/vihsd-visobert"
    assert fake_api.created == [
        ("someone/vihsd-visobert", {"repo_type": "model", "exist_ok": True, "private": False})
    ]
    upload = fake_api.uploaded[0]
    assert upload["repo_id"] == "someone/vihsd-visobert"
    assert upload["folder_path"] == str(bundle_dir)


def test_push_keeps_hate_text_out_of_the_hub_by_default(
    bundle_dir: Path, fake_api: FakeApi
) -> None:
    """demo_test_set.csv is real ViHSD validation text.

    The weights are worth publishing; the sample adds nothing the dataset lacks.
    """
    assert (bundle_dir / "demo_test_set.csv").is_file()
    publish.push_bundle_to_hub(bundle_dir, "someone/vihsd-visobert")
    ignore = cast("list[str]", fake_api.uploaded[0]["ignore_patterns"])
    assert "demo_test_set.csv" in ignore
    assert "*.zip" in ignore


def test_push_can_include_the_sample_when_asked(bundle_dir: Path, fake_api: FakeApi) -> None:
    publish.push_bundle_to_hub(bundle_dir, "someone/vihsd-visobert", include_sample=True)
    ignore = cast("list[str]", fake_api.uploaded[0]["ignore_patterns"])
    assert "demo_test_set.csv" not in ignore


def test_push_honours_private(bundle_dir: Path, fake_api: FakeApi) -> None:
    publish.push_bundle_to_hub(bundle_dir, "someone/vihsd-visobert", private=True)
    assert fake_api.created[0][1]["private"] is True


def test_push_refuses_something_that_is_not_a_bundle(
    tmp_path: Path, fake_api: FakeApi
) -> None:
    with pytest.raises(DemoBundleError):
        publish.push_bundle_to_hub(tmp_path / "nope", "someone/vihsd-visobert")
    assert fake_api.created == []
    assert fake_api.uploaded == []


# ------------------------------------------------------------- space assembly


def test_build_space_dir_lays_out_a_runnable_app(tmp_path: Path) -> None:
    space_dir = publish.build_space_dir(tmp_path / "space", "someone/vihsd-visobert")
    for name in ("app.py", "requirements.txt", "README.md"):
        assert (space_dir / name).is_file(), f"Space is missing {name}"
    for name in publish.SPACE_MODULES:
        assert (space_dir / "src" / "vihate" / name).is_file(), f"Space is missing {name}"


def test_space_requirements_cover_the_serving_stack(tmp_path: Path) -> None:
    requirements = (
        publish.build_space_dir(tmp_path / "space", "someone/vihsd-visobert") / "requirements.txt"
    ).read_text(encoding="utf-8")
    for package in ("torch", "transformers", "huggingface-hub", "pandas", "gradio"):
        assert package in requirements


def test_space_app_bakes_in_the_model_repo_and_port(tmp_path: Path) -> None:
    space_dir = publish.build_space_dir(
        tmp_path / "space", "someone/vihsd-visobert", SpaceSpec(app_port=7861)
    )
    source = (space_dir / "app.py").read_text(encoding="utf-8")
    baked = {
        node.targets[0].id: node.value.value  # type: ignore[attr-defined]
        for node in ast.parse(source).body
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
    }
    assert baked["MODEL_REPO"] == "someone/vihsd-visobert"
    assert 'host="0.0.0.0"' in source  # a Space is reached from outside the container
    assert "port=7861" in source
    assert "serve(ServeConfig(" in source


def test_space_app_puts_the_shipped_package_on_the_path(tmp_path: Path) -> None:
    source = (
        publish.build_space_dir(tmp_path / "space", "someone/vihsd-visobert") / "app.py"
    ).read_text(encoding="utf-8")
    assert "sys.path.insert" in source
    assert "from vihate.serve_demo import ServeConfig, serve" in source


def test_space_readme_declares_the_gradio_sdk(tmp_path: Path) -> None:
    readme = publish.build_space_dir(
        tmp_path / "space", "someone/vihsd-visobert", SpaceSpec(title="ViHSD Demo")
    ) / "README.md"
    text = readme.read_text(encoding="utf-8")
    head = text.split("---")[1]
    assert "sdk: gradio" in head
    assert f"sdk_version: {publish.SPACE_DEFAULT_SDK_VERSION}" in head
    assert f"app_port: {publish.SPACE_DEFAULT_PORT}" in head
    assert "title: ViHSD Demo" in head
    assert "someone/vihsd-visobert" in text


def test_space_readme_keeps_the_content_warning(tmp_path: Path) -> None:
    body = (
        publish.build_space_dir(tmp_path / "space", "someone/vihsd-visobert") / "README.md"
    ).read_text(encoding="utf-8")
    assert "Content warning" in body


def test_build_space_dir_overwrites_a_stale_layout(tmp_path: Path) -> None:
    target = tmp_path / "space"
    publish.build_space_dir(target, "a/b")
    (target / "leftover.py").write_text("# from an earlier publish", encoding="utf-8")
    publish.build_space_dir(target, "a/b")
    assert not (target / "leftover.py").exists()


def test_build_space_dir_fails_loudly_without_the_app_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(publish, "SPACE_MODULES", (*publish.SPACE_MODULES, "gone.py"))
    with pytest.raises(publish.MissingSpaceSourceError, match=r"gone\.py"):
        publish.build_space_dir(tmp_path / "space", "a/b")


def test_create_demo_space_uploads_to_a_space_repo(
    tmp_path: Path, fake_api: FakeApi, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub_package_dir(monkeypatch, tmp_path)
    url = publish.create_demo_space("someone/vihsd-demo", "someone/vihsd-visobert")
    assert url == "https://huggingface.co/spaces/someone/vihsd-demo"
    assert fake_api.created == [
        (
            "someone/vihsd-demo",
            {
                "repo_type": "space",
                "space_sdk": publish.SPACE_SDK,
                "exist_ok": True,
                "private": False,
            },
        )
    ]
    assert fake_api.uploaded[0]["repo_id"] == "someone/vihsd-demo"
    assert fake_api.uploaded[0]["repo_type"] == "space"


def test_create_demo_space_honours_private(
    tmp_path: Path, fake_api: FakeApi, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub_package_dir(monkeypatch, tmp_path)
    publish.create_demo_space("s/d", "m/r", SpaceSpec(private=True))
    assert fake_api.created[0][1]["private"] is True


# ------------------------------------------------------------------------- cli


def test_cli_publish_model_wires_through(
    bundle_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vihate.cli import app

    seen: dict[str, object] = {}

    def record(*args: object, **kwargs: object) -> str:
        seen["args"] = args
        seen.update(kwargs)
        return "https://huggingface.co/x/y"

    monkeypatch.setattr(publish, "push_bundle_to_hub", record)
    result = runner.invoke(
        app,
        ["publish-model", "--bundle", str(bundle_dir), "--repo", "x/y",
         "--private", "--include-sample"],
    )
    assert result.exit_code == 0, result.output
    assert "https://huggingface.co/x/y" in result.output
    assert seen["private"] is True
    assert seen["include_sample"] is True


def test_cli_publish_space_wires_through(monkeypatch: pytest.MonkeyPatch) -> None:
    from vihate.cli import app

    seen: dict[str, object] = {}

    def record(space_id: str, model_repo_id: str, spec: SpaceSpec | None = None) -> str:
        seen.update({"space_id": space_id, "model_repo_id": model_repo_id, "spec": spec})
        return "https://huggingface.co/spaces/x/y"

    monkeypatch.setattr(publish, "create_demo_space", record)
    result = runner.invoke(
        app, ["publish-space", "--space", "x/y", "--model", "x/z", "--app-port", "7999"]
    )
    assert result.exit_code == 0, result.output
    assert seen["space_id"] == "x/y"
    assert seen["model_repo_id"] == "x/z"
    spec = seen["spec"]
    assert isinstance(spec, SpaceSpec)
    assert spec.app_port == 7999
    assert "Build log" in result.output


def test_bundle_metadata_is_not_reported_by_the_helper_anymore(
    bundle_dir: Path, fake_api: FakeApi
) -> None:
    """push_bundle_to_hub returns just the URL; the CLI owns user-facing wording."""
    from vihate.demo_bundle import load_demo_config

    config = load_demo_config(bundle_dir)
    assert config.backbone == BACKBONE
    assert config.labels == LABELS
    url = publish.push_bundle_to_hub(bundle_dir, "someone/vihsd-visobert")
    assert "\n" not in url
    assert fake_api.created  # the stub answered, not the network


def test_publish_model_verifies_the_bundle_before_pushing(
    bundle_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bundle that cannot serve must not reach the Hub."""
    from vihate import serve_demo
    from vihate.cli import app

    pushed: list[object] = []

    def refuse(source: object) -> object:
        message = "probe failed"
        raise DemoBundleError(message, source=str(source))

    def record_push(*args: object, **_kwargs: object) -> str:
        pushed.append(args)
        return "https://huggingface.co/x/y"

    monkeypatch.setattr(serve_demo, "verify_bundle", refuse)
    monkeypatch.setattr(publish, "push_bundle_to_hub", record_push)
    result = runner.invoke(
        app, ["publish-model", "--bundle", str(bundle_dir), "--repo", "x/y"]
    )
    assert result.exit_code != 0
    assert pushed == []
