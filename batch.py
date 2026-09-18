"""Read an uploaded comment file and shape prediction results into a table.

Pure pandas/stdlib: no torch, no Gradio, so it is trivial to test.
"""

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

MAX_ROWS = 500
TEXT_COLUMNS = ("free_text", "text", "comment")
TABLE_SEPARATORS = {".csv": ",", ".tsv": "\t"}


class BatchFileError(ValueError):
    """The uploaded file cannot be analyzed; the message is safe to show to the user."""


def read_comments(path: str | Path) -> list[str]:
    """Return the non-empty comments in a .txt (one per line) or .csv/.tsv file."""
    path = Path(path)
    suffix = path.suffix.lower()
    try:
        if suffix in TABLE_SEPARATORS:
            table = pd.read_csv(path, sep=TABLE_SEPARATORS[suffix], encoding="utf-8-sig")
            column = next((name for name in TEXT_COLUMNS if name in table.columns), None)
            values = table[column or table.columns[0]].dropna().astype(str)
            lines = values.tolist()
        elif suffix == ".txt":
            lines = path.read_text(encoding="utf-8-sig").splitlines()
        else:
            message = f"Unsupported file type '{suffix}'. Upload a .txt, .csv or .tsv file."
            raise BatchFileError(message)
    except (
        OSError,
        UnicodeDecodeError,
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
    ) as exc:
        message = f"Could not read {path.name}: {exc}"
        raise BatchFileError(message) from exc

    comments = [line.strip() for line in lines if line.strip()]
    if not comments:
        message = f"{path.name} has no comments to analyze."
        raise BatchFileError(message)
    return comments


def top_label(scores: dict[str, float]) -> str:
    """Return the label with the highest probability."""
    return max(scores, key=lambda label: scores[label])


def results_table(texts: Sequence[str], scores: Sequence[dict[str, float]]) -> pd.DataFrame:
    """One row per comment: the text, the predicted class and every class probability."""
    rows = []
    for text, text_scores in zip(texts, scores, strict=True):
        row: dict[str, object] = {"text": text, "predicted_class": top_label(text_scores)}
        row.update({f"P({label})": round(prob, 4) for label, prob in text_scores.items()})
        rows.append(row)
    return pd.DataFrame(rows)
