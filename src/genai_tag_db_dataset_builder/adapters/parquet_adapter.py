"""Parquet_Adapter for reading Parquet files from HuggingFace datasets.

This adapter handles Parquet file reading for datasets like deepghs/site_tags
and isek-ai/danbooru-wiki-2024.
"""

from pathlib import Path

import polars as pl

from .base_adapter import BaseAdapter


class Parquet_Adapter(BaseAdapter):
    """Parquetファイル用アダプタ（deepghs, danbooru-wiki）.

    Args:
        file_path: Parquetファイルのパス
    """

    def __init__(self, file_path: Path | str) -> None:
        """Initialize adapter.

        Args:
            file_path: Path to Parquet file

        Raises:
            FileNotFoundError: Parquetファイルが存在しない場合
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {self.file_path}")

    def read(self) -> pl.DataFrame:
        """Parquetファイルを読み込み.

        Returns:
            正規化されたPolars DataFrame

        Raises:
            ValueError: Parquet読み込みに失敗した場合
        """
        try:
            df = pl.read_parquet(self.file_path)
            return df

        except Exception as e:
            raise ValueError(f"Failed to read Parquet: {self.file_path}") from e

    def validate(self, df: pl.DataFrame) -> bool:
        """データ整合性検証.

        Args:
            df: 検証対象のDataFrame

        Returns:
            検証成功の場合True
        """
        if df.is_empty():
            return False

        return True

    def repair(self, df: pl.DataFrame) -> pl.DataFrame:
        """データ修復.

        Args:
            df: 修復対象のDataFrame

        Returns:
            修復されたDataFrame（Parquet は基本的に修復不要）
        """
        return df
