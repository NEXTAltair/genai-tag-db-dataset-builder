"""Unit tests for JSON adapter."""

import json
from pathlib import Path

import polars as pl
import pytest

from genai_tag_db_dataset_builder.adapters.json_adapter import JSON_Adapter


class TestJSON_Adapter:
    """JSON_Adapterのテスト."""

    def test_init_with_nonexistent_file(self, tmp_path: Path) -> None:
        """存在しないファイルでの初期化."""
        json_path = tmp_path / "nonexistent.json"
        with pytest.raises(FileNotFoundError):
            JSON_Adapter(json_path)

    def test_read_simple_json(self, tmp_path: Path) -> None:
        """シンプルなJSON読み込み."""
        json_path = tmp_path / "test.json"
        test_data = [
            {"source_tag": "witch", "deprecated_tags": "sorceress"},
            {"source_tag": "mage", "deprecated_tags": "wizard"},
        ]
        json_path.write_text(json.dumps(test_data), encoding="utf-8")

        adapter = JSON_Adapter(json_path)
        df = adapter.read()

        assert len(df) == 2
        assert "source_tag" in df.columns
        assert "deprecated_tags" in df.columns
        assert df["source_tag"].to_list() == ["witch", "mage"]

    def test_read_single_object_json(self, tmp_path: Path) -> None:
        """単一オブジェクトJSON（リストでない）の読み込み."""
        json_path = tmp_path / "test.json"
        test_data = {"source_tag": "witch", "deprecated_tags": "sorceress"}
        json_path.write_text(json.dumps(test_data), encoding="utf-8")

        adapter = JSON_Adapter(json_path)
        df = adapter.read()

        # 単一オブジェクトもDataFrameに変換される
        assert len(df) == 1
        assert df["source_tag"][0] == "witch"

    def test_column_normalization_tag_to_source_tag(self, tmp_path: Path) -> None:
        """tag列→source_tag列への正規化."""
        json_path = tmp_path / "test.json"
        test_data = [
            {"tag": "witch_hat", "deprecated_tags": "sorceress"},
            {"tag": "mage_staff", "deprecated_tags": "wizard"},
        ]
        json_path.write_text(json.dumps(test_data), encoding="utf-8")

        adapter = JSON_Adapter(json_path)
        df = adapter.read()

        # tag列がsource_tagにリネームされている
        assert "source_tag" in df.columns
        assert "tag" not in df.columns
        assert df["source_tag"].to_list() == ["witch_hat", "mage_staff"]

    def test_validate_valid_dataframe(self, tmp_path: Path) -> None:
        """正常なDataFrameの検証."""
        json_path = tmp_path / "test.json"
        test_data = [{"source_tag": "witch"}]
        json_path.write_text(json.dumps(test_data), encoding="utf-8")

        adapter = JSON_Adapter(json_path)
        df = adapter.read()

        assert adapter.validate(df) is True

    def test_validate_empty_dataframe(self) -> None:
        """空のDataFrameの検証."""
        adapter = JSON_Adapter(__file__)

        empty_df = pl.DataFrame()
        assert adapter.validate(empty_df) is False

    def test_validate_enforces_source_tag_column(self) -> None:
        """source_tag列の必須検証."""
        adapter = JSON_Adapter(__file__)

        # source_tag列が無いDataFrame
        df_without_source_tag = pl.DataFrame({"other_column": ["value1", "value2"]})

        assert adapter.validate(df_without_source_tag) is False

    def test_repair_returns_unchanged(self, tmp_path: Path) -> None:
        """repair()がDataFrameをそのまま返すことを確認."""
        json_path = tmp_path / "test.json"
        test_data = [{"source_tag": "witch"}]
        json_path.write_text(json.dumps(test_data), encoding="utf-8")

        adapter = JSON_Adapter(json_path)
        df = adapter.read()
        repaired_df = adapter.repair(df)

        assert repaired_df.equals(df)
