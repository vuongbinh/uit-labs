"""Command line entry point."""

from pathlib import Path
from typing import Annotated, Literal

import typer

from vihate.classical import ClassicalConfig, run_classical_cv
from vihate.data import load_vihsd_examples
from vihate.publish import SPACE_DEFAULT_PORT, SPACE_DEFAULT_SDK_VERSION, SPACE_DEFAULT_TITLE
from vihate.reporting import persist_fold_results
from vihate.transformer_cv import TransformerConfig, run_transformer_cv

# serve_demo and publish's heavy work are imported inside the commands below, so
# `vihate --help` stays fast and works before the ML extras are installed.

app = typer.Typer(no_args_is_help=True)


@app.callback()
def _callback() -> None:
    """Vietnamese hate-speech (ViHSD) training and cross-validation pipeline."""


@app.command()
def run(
    experiment: Annotated[Literal["classical", "transformer"], typer.Option()] = "classical",
    out_dir: Annotated[Path, typer.Option()] = Path("outputs/run"),
    split: Annotated[str, typer.Option()] = "train",
    folds: Annotated[int, typer.Option()] = 5,
    seed: Annotated[int, typer.Option()] = 13,
    sample_size: Annotated[int | None, typer.Option()] = None,
    classical_model: Annotated[Literal["logreg", "svm"], typer.Option()] = "logreg",
    model_name: Annotated[str, typer.Option()] = "vinai/phobert-base",
    epochs: Annotated[float, typer.Option()] = 3.0,
    batch_size: Annotated[int, typer.Option()] = 16,
    grad_accum_steps: Annotated[int, typer.Option()] = 1,
    learning_rate: Annotated[float, typer.Option()] = 2e-5,
    max_length: Annotated[int, typer.Option()] = 160,
    warmup_ratio: Annotated[float, typer.Option()] = 0.1,
    optim: Annotated[str, typer.Option()] = "adamw_torch",
    gradient_checkpointing: Annotated[bool, typer.Option()] = False,
    freeze_embeddings: Annotated[bool, typer.Option()] = False,
) -> None:
    """Run a ViHSD cross-validation experiment."""
    examples = load_vihsd_examples(split=split, sample_size=sample_size)
    match experiment:
        case "classical":
            fold_results = run_classical_cv(
                examples,
                ClassicalConfig(model=classical_model, folds=folds, seed=seed),
                out_dir,
            )
        case "transformer":
            fold_results = run_transformer_cv(
                examples,
                TransformerConfig(
                    model_name=model_name,
                    folds=folds,
                    seed=seed,
                    epochs=epochs,
                    batch_size=batch_size,
                    grad_accum_steps=grad_accum_steps,
                    learning_rate=learning_rate,
                    max_length=max_length,
                    warmup_ratio=warmup_ratio,
                    optim=optim,
                    gradient_checkpointing=gradient_checkpointing,
                    freeze_embeddings=freeze_embeddings,
                ),
                out_dir,
            )
    persist_fold_results(out_dir, fold_results)


@app.command()
def demo(
    model: Annotated[
        str, typer.Option(help="Hub repo id or local demo bundle directory written by Part 7B.")
    ],
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[
        int, typer.Option(help="Mirrors serve_demo.DEFAULT_PORT.")
    ] = SPACE_DEFAULT_PORT,
    share: Annotated[
        bool, typer.Option("--share/--no-share", help="Public tunnel URL; expires in about a week.")
    ] = False,
    device: Annotated[str | None, typer.Option(help="cuda | cpu (default: auto)")] = None,
    out_dir: Annotated[
        Path | None, typer.Option(help="Where batch CSV exports are written (default: a temp dir).")
    ] = None,
) -> None:
    """Serve the Gradio demo from a persisted bundle -- no training, no notebook."""
    from vihate.serve_demo import HateSpeechPredictor, ServeConfig, launch_app

    predictor = HateSpeechPredictor(model, device=device, output_dir=out_dir)
    typer.echo(
        f"ready — {predictor.config.backbone} on {predictor.device}, "
        f"labels {list(predictor.labels)}, t* {predictor.t_star:.3f}"
    )
    config = ServeConfig(
        source=model, host=host, port=port, share=share, device=device, output_dir=out_dir
    )
    launch_app(predictor, config)


@app.command(name="publish-model")
def publish_model(
    bundle: Annotated[Path, typer.Option(help="Demo bundle directory written by Part 7B.")],
    repo: Annotated[str, typer.Option(help="Target repo id, e.g. yourname/vihsd-visobert.")],
    private: Annotated[bool, typer.Option("--private/--public")] = False,
    include_sample: Annotated[
        bool,
        typer.Option(
            "--include-sample/--no-sample",
            help="Also upload demo_test_set.csv (real ViHSD text containing slurs).",
        ),
    ] = False,
    token: Annotated[
        str | None, typer.Option(help="HF write token; defaults to the logged-in one.")
    ] = None,
) -> None:
    """Upload a demo bundle to a Hugging Face model repo."""
    from vihate.publish import push_bundle_to_hub

    url = push_bundle_to_hub(
        bundle, repo, private=private, include_sample=include_sample, token=token
    )
    typer.echo(f"pushed {bundle} -> {url}")


@app.command(name="publish-space")
def publish_space(
    space: Annotated[str, typer.Option(help="Target space id, e.g. yourname/vihsd-demo.")],
    model: Annotated[str, typer.Option(help="Model repo id the Space loads at startup.")],
    title: Annotated[str, typer.Option()] = SPACE_DEFAULT_TITLE,
    sdk_version: Annotated[str, typer.Option()] = SPACE_DEFAULT_SDK_VERSION,
    app_port: Annotated[int, typer.Option()] = SPACE_DEFAULT_PORT,
    private: Annotated[bool, typer.Option("--private/--public")] = False,
    token: Annotated[
        str | None, typer.Option(help="HF write token; defaults to the logged-in one.")
    ] = None,
) -> None:
    """Create or update a Gradio Space that serves the published model."""
    from vihate.publish import SpaceSpec, create_demo_space

    spec = SpaceSpec(
        title=title, sdk_version=sdk_version, app_port=app_port, private=private, token=token
    )
    url = create_demo_space(space, model, spec)
    typer.echo(f"space files pushed -> {url}")
    typer.echo("the Space builds on first push; watch its Build log, then use the App tab")


if __name__ == "__main__":
    app()
