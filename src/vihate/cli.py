"""Command line entry point."""

from pathlib import Path
from typing import Annotated, Literal

import typer

from vihate.classical import ClassicalConfig, run_classical_cv
from vihate.data import load_vihsd_examples
from vihate.reporting import persist_fold_results
from vihate.transformer_cv import TransformerConfig, run_transformer_cv

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


if __name__ == "__main__":
    app()
