"""The conventional app.py entrypoint: module contract and a real HTTP boot."""

import importlib.util
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import cast

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
APP = REPO_ROOT / "app.py"


def load_app_module(bundle: Path) -> object:
    """Import app.py by path with DEMO_MODEL pointed at a bundle."""
    os.environ["DEMO_MODEL"] = str(bundle)
    spec = importlib.util.spec_from_file_location("vihsd_app_under_test", APP)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        _, port = probe.getsockname()
    return int(port)


def wait_for_port(port: int, proc: "subprocess.Popen[str]", budget: float = 120.0) -> bool:
    deadline = time.time() + budget
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def kill_tree(proc: "subprocess.Popen[str]") -> str:
    """Kill the whole process group; a reload child otherwise keeps the pipe open."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        proc.terminate()
    try:
        out, _ = proc.communicate(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        out, _ = proc.communicate(timeout=15)
    return out or ""


def test_app_py_exists_at_the_repo_root() -> None:
    assert APP.is_file(), "app.py must live at the repo root for `gradio app.py`"


def test_module_exposes_demo_as_a_blocks(bundle_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`gradio app.py` resolves --demo-name demo, so that name must hold the Blocks."""
    import gradio as gr

    monkeypatch.setenv("DEMO_MODEL", str(bundle_dir))
    module = load_app_module(bundle_dir)
    assert isinstance(module.demo, gr.Blocks)  # type: ignore[attr-defined]
    assert module.predictor.config.labels == ("CLEAN", "OFFENSIVE", "HATE")  # type: ignore[attr-defined]


def test_python_app_py_serves_over_http(bundle_dir: Path) -> None:
    port = free_port()
    proc = subprocess.Popen(  # noqa: S603
        [sys.executable, str(APP)],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
        env={
            **os.environ,
            "DEMO_MODEL": str(bundle_dir),
            "DEMO_PORT": str(port),
            "GRADIO_ANALYTICS_ENABLED": "False",
        },
    )
    try:
        assert wait_for_port(port, proc), "app.py never listened"
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=30) as response:
            body = response.read().decode("utf-8")
        assert response.status == 200
        assert "Vietnamese Hate Speech Detection" in body

        from gradio_client import Client

        client = Client(f"http://127.0.0.1:{port}/", verbose=False)
        api_info = cast("dict[str, dict[str, object]]", client.view_api(return_format="dict"))
        named = api_info.get("named_endpoints", {})
        assert sorted(named) == ["/predict_batch", "/predict_detailed"]
    finally:
        kill_tree(proc)


def test_missing_bundle_fails_loudly(tmp_path: Path) -> None:
    proc = subprocess.run(  # noqa: S603
        [sys.executable, str(APP)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env={**os.environ, "DEMO_MODEL": str(tmp_path / "no-such-bundle")},
    )
    assert proc.returncode != 0
    assert "DemoBundleError" in proc.stderr
