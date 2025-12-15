"""Unit tests for CSV_Adapter."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from genai_tag_db_dataset_builder.adapters.csv_adapter import CSV_Adapter
from genai_tag_db_dataset_builder.core.exceptions import NormalizedSourceSkipError
from genai_tag_db_dataset_builder.core.overrides import ColumnTypeOverrides


class TestCSVAdapter:
    def test_init_with_nonexistent_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            CSV_Adapter("/nonexistent/path/test.csv")

    def test_repair_derpibooru_adds_format_id(self) -> None:
        df = pl.DataFrame(
            {
                "source_tag": ["tag1", "tag2"],
                "type_id": [1, 2],
                "count": [100, 200],
            }
        )
        adapter = CSV_Adapter(__file__, repair_mode="derpibooru")
        repaired = adapter._repair_derpibooru(df)
        assert "format_id" in repaired.columns
        assert repaired["format_id"].to_list() == [3, 3]

    def test_repair_dataset_rising_v2_drops_mostly_null_columns(self) -> None:
        df = pl.DataFrame(
            {
                "tag": ["tag1", "tag2", "tag3"],
                "deprecated_tags": ["old1", "old2", "old3"],
                "empty_col1": [None, None, None],
                "empty_col2": [None, None, "value"],
            }
        )
        adapter = CSV_Adapter(__file__, repair_mode="dataset_rising_v2")
        repaired = adapter._repair_dataset_rising_v2(df)
        assert "tag" in repaired.columns
        assert "deprecated_tags" in repaired.columns
        assert "empty_col1" not in repaired.columns
        assert "empty_col2" in repaired.columns

    def test_repair_english_dict_fomat_id_typo(self) -> None:
        df = pl.DataFrame(
            {
                "source_tag": ["tag1", "tag2"],
                "type_id": [1, 2],
                "fomat_id": [1, 1],
            }
        )
        adapter = CSV_Adapter(__file__, repair_mode="english_dict")
        repaired = adapter._repair_english_dict(df)
        assert "format_id" in repaired.columns
        assert "fomat_id" not in repaired.columns

    def test_validate_valid_dataframe(self) -> None:
        df = pl.DataFrame({"source_tag": ["tag1", "tag2"], "type_id": [1, 2]})
        adapter = CSV_Adapter(__file__)
        assert adapter.validate(df) is True

    def test_validate_requires_source_tag_column(self) -> None:
        df = pl.DataFrame({"type_id": [1, 2], "count": [100, 200]})
        adapter = CSV_Adapter(__file__)
        assert adapter.validate(df) is False

    def test_validate_empty_dataframe(self) -> None:
        adapter = CSV_Adapter(__file__)
        assert adapter.validate(pl.DataFrame()) is False

    def test_column_normalization_tag_to_source_tag_and_lowercase(self) -> None:
        df = pl.DataFrame({"tag": ["Witch", "Spiked_Collar"], "type_id": [4, 0]})
        # SOURCE判定させるため override を追加
        overrides = ColumnTypeOverrides({str(__file__): {"tag": "source"}})
        adapter = CSV_Adapter(__file__, overrides=overrides)
        normalized = adapter._normalize_columns(df)
        assert "source_tag" in normalized.columns
        assert "tag" not in normalized.columns
        assert normalized["source_tag"].to_list() == ["witch", "spiked_collar"]


    def test_normalized_source_raises_skip_error(self, tmp_path: Path) -> None:
        """NORMALIZED判定されたソースはスキップされる."""
        csv_path = tmp_path / "normalized.csv"
        csv_path.write_text("tag,type_id\nwitch hat,4\nmage staff,4\n", encoding="utf-8")
        adapter = CSV_Adapter(csv_path)
        with pytest.raises(NormalizedSourceSkipError) as exc_info:
            adapter.read()
        assert exc_info.value.decision == "normalized"
        assert exc_info.value.file_path == str(csv_path)

    def test_unknown_source_raises_skip_error(self, tmp_path: Path) -> None:
        """UNKNOWN判定されたソースはスキップされる."""
        csv_path = tmp_path / "unknown.csv"
        csv_path.write_text("tag,type_id\nwitch_hat,4\nlong hair,4\nred_eyes,4\n", encoding="utf-8")
        adapter = CSV_Adapter(csv_path)
        with pytest.raises(NormalizedSourceSkipError) as exc_info:
            adapter.read()
        assert exc_info.value.decision == "unknown"
        assert exc_info.value.file_path == str(csv_path)

    def test_normalized_source_with_override_does_not_skip(self, tmp_path: Path) -> None:
        """Override設定でNORMALIZED判定をバイパスできる."""
        csv_path = tmp_path / "normalized.csv"
        csv_path.write_text("tag,type_id\nwitch hat,4\nmage staff,4\n", encoding="utf-8")
        overrides = ColumnTypeOverrides({str(csv_path): {"tag": "source"}})
        adapter = CSV_Adapter(csv_path, overrides=overrides)
        df = adapter.read()
        assert "source_tag" in df.columns
        assert df["source_tag"].to_list() == ["witch hat", "mage staff"]

    def test_unknown_source_with_override_does_not_skip(self, tmp_path: Path) -> None:
        """Override設定でUNKNOWN判定をバイパスできる."""
        csv_path = tmp_path / "unknown.csv"
        csv_path.write_text("tag,type_id\nwitch_hat,4\nlong hair,4\n", encoding="utf-8")
        overrides = ColumnTypeOverrides({str(csv_path): {"tag": "source"}})
        adapter = CSV_Adapter(csv_path, overrides=overrides)
        df = adapter.read()
        assert "source_tag" in df.columns
        assert df["source_tag"].to_list() == ["witch_hat", "long hair"]

    def test_unknown_skip_exports_report(self, tmp_path: Path) -> None:
        """UNKNOWNスキップ時にレポートが出力される."""
        csv_path = tmp_path / "unknown.csv"
        csv_path.write_text("tag,type_id\nwitch_hat,4\nlong hair,4\n", encoding="utf-8")
        report_dir = tmp_path / "reports"
        adapter = CSV_Adapter(csv_path, unknown_report_dir=report_dir)
        with pytest.raises(NormalizedSourceSkipError):
            adapter.read()
        # レポートが出力されている
        assert any(report_dir.glob("*__unknown_tag.tsv"))

