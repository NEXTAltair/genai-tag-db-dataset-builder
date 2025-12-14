"""Unit tests for CSV_Adapter."""

from pathlib import Path

import polars as pl
import pytest

from genai_tag_db_dataset_builder.adapters.csv_adapter import CSV_Adapter


class TestCSV_Adapter:
    """CSV_Adapterのテスト."""

    def test_init_with_nonexistent_file(self) -> None:
        """存在しないファイルでの初期化テスト."""
        with pytest.raises(FileNotFoundError):
            CSV_Adapter("/nonexistent/path/test.csv")

    def test_repair_derpibooru(self) -> None:
        """derpibooru修復ロジックのテスト."""
        # format_id欠損DataFrameを作成
        df = pl.DataFrame({
            "source_tag": ["tag1", "tag2"],
            "type_id": [1, 2],
            "count": [100, 200],
        })

        adapter = CSV_Adapter(__file__, repair_mode="derpibooru")
        repaired = adapter._repair_derpibooru(df)

        # format_id=3が追加されていることを確認
        assert "format_id" in repaired.columns
        assert repaired["format_id"].to_list() == [3, 3]

    def test_repair_dataset_rising_v2(self) -> None:
        """dataset_rising_v2修復ロジックのテスト."""
        # 90%以上がnullの列を含むDataFrameを作成
        df = pl.DataFrame({
            "tag": ["tag1", "tag2", "tag3"],
            "deprecated_tags": ["old1", "old2", "old3"],
            "empty_col1": [None, None, None],
            "empty_col2": [None, None, "value"],
        })

        adapter = CSV_Adapter(__file__, repair_mode="dataset_rising_v2")
        repaired = adapter._repair_dataset_rising_v2(df)

        # empty_col1は削除、empty_col2は残る
        assert "tag" in repaired.columns
        assert "deprecated_tags" in repaired.columns
        assert "empty_col1" not in repaired.columns
        assert "empty_col2" in repaired.columns

    def test_repair_english_dict(self) -> None:
        """EnglishDictionary修復ロジックのテスト."""
        # fomat_id（typo）を含むDataFrameを作成
        df = pl.DataFrame({
            "source_tag": ["tag1", "tag2"],
            "type_id": [1, 2],
            "fomat_id": [1, 1],
        })

        adapter = CSV_Adapter(__file__, repair_mode="english_dict")
        repaired = adapter._repair_english_dict(df)

        # format_idにリネームされていることを確認
        assert "format_id" in repaired.columns
        assert "fomat_id" not in repaired.columns

    def test_validate_valid_dataframe(self) -> None:
        """有効なDataFrameの検証テスト."""
        df = pl.DataFrame({
            "source_tag": ["tag1", "tag2"],
            "type_id": [1, 2],
        })

        adapter = CSV_Adapter(__file__)
        assert adapter.validate(df) is True

    def test_validate_invalid_dataframe(self) -> None:
        """無効なDataFrameの検証テスト."""
        # source_tag列が存在しない
        df = pl.DataFrame({
            "type_id": [1, 2],
            "count": [100, 200],
        })

        adapter = CSV_Adapter(__file__)
        assert adapter.validate(df) is False

    def test_validate_empty_dataframe(self) -> None:
        """空DataFrameの検証テスト."""
        df = pl.DataFrame()

        adapter = CSV_Adapter(__file__)
        assert adapter.validate(df) is False
