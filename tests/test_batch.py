from pathlib import Path

import pandas as pd
import pytest

from batch import BatchFileError, read_comments, results_table, top_label


def test_txt_one_comment_per_line_skips_blanks(tmp_path: Path) -> None:
    path = tmp_path / "in.txt"
    path.write_text("xin chào\n\n  cảm ơn  \n", encoding="utf-8")
    assert read_comments(path) == ["xin chào", "cảm ơn"]


@pytest.mark.parametrize(
    ("suffix", "sep"),
    [(".csv", ","), (".tsv", "\t")],
)
def test_table_prefers_known_text_column(tmp_path: Path, suffix: str, sep: str) -> None:
    path = tmp_path / f"in{suffix}"
    path.write_text(f"id{sep}text{sep}label\n1{sep}xin chào{sep}0\n2{sep}cảm ơn{sep}0\n", "utf-8")
    assert read_comments(path) == ["xin chào", "cảm ơn"]


def test_table_falls_back_to_first_column(tmp_path: Path) -> None:
    path = tmp_path / "in.csv"
    path.write_text("noidung,other\nxin chào,1\n", encoding="utf-8")
    assert read_comments(path) == ["xin chào"]


def test_table_drops_missing_values(tmp_path: Path) -> None:
    path = tmp_path / "in.csv"
    path.write_text("text\nxin chào\n\ncảm ơn\n", encoding="utf-8")
    assert read_comments(path) == ["xin chào", "cảm ơn"]


def test_utf8_bom_is_ignored(tmp_path: Path) -> None:
    path = tmp_path / "in.csv"
    path.write_text("text\nxin chào\n", encoding="utf-8-sig")
    assert read_comments(path) == ["xin chào"]


@pytest.mark.parametrize("name", ["empty.txt", "empty.csv"])
def test_empty_file_raises(tmp_path: Path, name: str) -> None:
    path = tmp_path / name
    path.write_text("", encoding="utf-8")
    with pytest.raises(BatchFileError):
        read_comments(path)


def test_unsupported_suffix_raises(tmp_path: Path) -> None:
    path = tmp_path / "in.pdf"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(BatchFileError, match=r"\.pdf"):
        read_comments(path)


def test_non_utf8_bytes_raise_readable_error(tmp_path: Path) -> None:
    path = tmp_path / "in.txt"
    path.write_bytes(b"\xff\xfe\x00bad")
    with pytest.raises(BatchFileError, match=r"in\.txt"):
        read_comments(path)


def test_top_label_is_argmax() -> None:
    assert top_label({"CLEAN": 0.1, "OFFENSIVE": 0.2, "HATE": 0.7}) == "HATE"


def test_results_table_shape_and_rounding() -> None:
    scores = [
        {"CLEAN": 0.123456, "OFFENSIVE": 0.3, "HATE": 0.576544},
        {"CLEAN": 0.9, "OFFENSIVE": 0.05, "HATE": 0.05},
    ]
    table = results_table(["a", "b"], scores)
    assert list(table.columns) == [
        "text",
        "predicted_class",
        "P(CLEAN)",
        "P(OFFENSIVE)",
        "P(HATE)",
    ]
    assert table["predicted_class"].tolist() == ["HATE", "CLEAN"]
    assert table["P(CLEAN)"].tolist() == [0.1235, 0.9]
    assert isinstance(table, pd.DataFrame)
