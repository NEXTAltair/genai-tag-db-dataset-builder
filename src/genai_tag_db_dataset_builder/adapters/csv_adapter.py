"""CSV_Adapter for reading and repairing CSV files.

This adapter handles CSV file reading with automatic repair logic
for broken CSV files (extra commas, missing columns, etc.).
"""

from pathlib import Path

import polars as pl

from .base_adapter import BaseAdapter


class CSV_Adapter(BaseAdapter):
    """汎用CSVアダプタ（壊れたCSVの修復機能付き）.

    Args:
        file_path: CSVファイルのパス
        repair_mode: 修復モード（"derpibooru", "dataset_rising_v2", "english_dict"）
    """

    def __init__(self, file_path: Path | str, repair_mode: str | None = None) -> None:
        """Initialize adapter.

        Args:
            file_path: Path to CSV file
            repair_mode: Repair mode for specific broken CSV files

        Raises:
            FileNotFoundError: CSVファイルが存在しない場合
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.file_path}")

        self.repair_mode = repair_mode

    def read(self) -> pl.DataFrame:
        """CSVファイルを読み込み.

        Returns:
            正規化されたPolars DataFrame

        Raises:
            ValueError: CSV読み込みに失敗した場合
        """
        try:
            df = pl.read_csv(
                self.file_path,
                ignore_errors=True,
                truncate_ragged_lines=True,
            )
        except Exception as e:
            raise ValueError(f"Failed to read CSV: {self.file_path}") from e

        # 修復モード適用
        if self.repair_mode:
            df = self.repair(df)

        return df

    def validate(self, df: pl.DataFrame) -> bool:
        """データ整合性検証.

        Args:
            df: 検証対象のDataFrame

        Returns:
            検証成功の場合True
        """
        if df.is_empty():
            return False

        # source_tag列の存在確認
        if "source_tag" not in df.columns and "tag" not in df.columns:
            return False

        return True

    def repair(self, df: pl.DataFrame) -> pl.DataFrame:
        """壊れたデータの修復.

        Args:
            df: 修復対象のDataFrame

        Returns:
            修復されたDataFrame
        """
        if self.repair_mode == "derpibooru":
            return self._repair_derpibooru(df)
        elif self.repair_mode == "dataset_rising_v2":
            return self._repair_dataset_rising_v2(df)
        elif self.repair_mode == "english_dict":
            return self._repair_english_dict(df)
        else:
            return df

    def _repair_derpibooru(self, df: pl.DataFrame) -> pl.DataFrame:
        """derpibooru.csv修復: format_id欠損を補完.

        Args:
            df: 修復対象のDataFrame

        Returns:
            修復されたDataFrame
        """
        if "format_id" not in df.columns:
            df = df.with_columns(pl.lit(3).alias("format_id"))
        return df

    def _repair_dataset_rising_v2(self, df: pl.DataFrame) -> pl.DataFrame:
        """dataset_rising_v2.csv修復: 余計なカンマで生成された空列を削除.

        Args:
            df: 修復対象のDataFrame

        Returns:
            修復されたDataFrame
        """
        # 90%以上がnullの列を削除
        cols_to_keep = []
        for col in df.columns:
            null_ratio = df[col].is_null().sum() / len(df)
            if null_ratio < 0.9:
                cols_to_keep.append(col)

        return df.select(cols_to_keep)

    def _repair_english_dict(self, df: pl.DataFrame) -> pl.DataFrame:
        """EnglishDictionary.csv修復: fomat_id→format_idリネーム.

        Args:
            df: 修復対象のDataFrame

        Returns:
            修復されたDataFrame
        """
        if "fomat_id" in df.columns:
            df = df.rename({"fomat_id": "format_id"})
        return df
