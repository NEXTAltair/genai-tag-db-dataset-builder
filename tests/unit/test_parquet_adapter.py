"""Unit tests for Parquet adapter."""

from pathlib import Path

import polars as pl
import pytest

from genai_tag_db_dataset_builder.adapters.parquet_adapter import Parquet_Adapter


class TestParquet_Adapter:
    """Parquet_Adapterのテスト."""

    def test_init_with_nonexistent_file(self, tmp_path: Path) -> None:
        """存在しないファイルでの初期化."""
        parquet_path = tmp_path / "nonexistent.parquet"
        with pytest.raises(FileNotFoundError):
            Parquet_Adapter(parquet_path)

    def test_read_parquet(self, tmp_path: Path) -> None:
        """Parquetファイル読み込み."""
        parquet_path = tmp_path / "test.parquet"
        test_df = pl.DataFrame(
            {
                "source_tag": ["witch", "mage"],
                "deprecated_tags": ["sorceress", "wizard"],
            }
        )
        test_df.write_parquet(parquet_path)

        adapter = Parquet_Adapter(parquet_path)
        df = adapter.read()

        assert len(df) == 2
        assert "source_tag" in df.columns
        assert "deprecated_tags" in df.columns
        assert df["source_tag"].to_list() == ["witch", "mage"]

    def test_column_normalization_tag_to_source_tag(self, tmp_path: Path) -> None:
        """tag列→source_tag列への正規化."""
        parquet_path = tmp_path / "test.parquet"
        test_df = pl.DataFrame(
            {
                "tag": ["witch_hat", "mage_staff"],
                "deprecated_tags": ["sorceress", "wizard"],
            }
        )
        test_df.write_parquet(parquet_path)

        adapter = Parquet_Adapter(parquet_path)
        df = adapter.read()

        # tag列がsource_tagにリネームされている
        assert "source_tag" in df.columns
        assert "tag" not in df.columns
        assert df["source_tag"].to_list() == ["witch_hat", "mage_staff"]

    def test_validate_valid_dataframe(self, tmp_path: Path) -> None:
        """正常なDataFrameの検証."""
        parquet_path = tmp_path / "test.parquet"
        test_df = pl.DataFrame({"source_tag": ["witch"]})
        test_df.write_parquet(parquet_path)

        adapter = Parquet_Adapter(parquet_path)
        df = adapter.read()

        assert adapter.validate(df) is True

    def test_validate_empty_dataframe(self) -> None:
        """空のDataFrameの検証."""
        adapter = Parquet_Adapter(__file__)

        empty_df = pl.DataFrame()
        assert adapter.validate(empty_df) is False

    def test_validate_enforces_source_tag_column(self) -> None:
        """source_tag列の必須検証."""
        adapter = Parquet_Adapter(__file__)

        # source_tag列が無いDataFrame
        df_without_source_tag = pl.DataFrame({"other_column": ["value1", "value2"]})

        assert adapter.validate(df_without_source_tag) is False

    def test_repair_returns_unchanged(self, tmp_path: Path) -> None:
        """repair()がDataFrameをそのまま返すことを確認."""
        parquet_path = tmp_path / "test.parquet"
        test_df = pl.DataFrame({"source_tag": ["witch"]})
        test_df.write_parquet(parquet_path)

        adapter = Parquet_Adapter(parquet_path)
        df = adapter.read()
        repaired_df = adapter.repair(df)

        assert repaired_df.equals(df)
