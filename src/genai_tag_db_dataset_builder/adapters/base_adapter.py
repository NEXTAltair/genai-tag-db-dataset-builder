"""Base adapter class for data source integration.

This module defines the abstract base class for all data source adapters,
ensuring consistent interface for reading, validating, and repairing data.
"""

from abc import ABC, abstractmethod

import polars as pl

# 標準列名への正規化
STANDARD_COLUMNS = {
    "source_tag": str,  # 必須
    "tag": str,  # tagがあればsource_tagとして扱う
    "type_id": int | None,
    "format_id": int | None,
    "count": int | None,
    "deprecated_tags": str | None,
    "japanese": str | None,
    "translation": str | None,
}


class BaseAdapter(ABC):
    """入力ソースアダプタ基底クラス.

    全てのデータソースアダプタはこのクラスを継承し、
    read(), validate(), repair()メソッドを実装する必要があります。
    """

    @abstractmethod
    def read(self) -> pl.DataFrame | dict[str, pl.DataFrame]:
        """ファイルを読み込んでPolars DataFrameに正規化.

        Returns:
            正規化されたPolars DataFrame（STANDARD_COLUMNS準拠）
            または複数テーブルの辞書（Tags_v4_Adapterの場合）

        Raises:
            FileNotFoundError: ファイルが存在しない場合
            ValueError: データ形式が不正な場合
        """
        pass

    @abstractmethod
    def validate(self, df: pl.DataFrame) -> bool:
        """データ整合性検証.

        Args:
            df: 検証対象のDataFrame

        Returns:
            検証成功の場合True、失敗の場合False
        """
        pass

    @abstractmethod
    def repair(self, df: pl.DataFrame) -> pl.DataFrame:
        """壊れたデータの修復.

        Args:
            df: 修復対象のDataFrame

        Returns:
            修復されたDataFrame
        """
        pass
