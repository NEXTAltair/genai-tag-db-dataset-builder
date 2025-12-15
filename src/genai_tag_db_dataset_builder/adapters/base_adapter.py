"""データソース統合用アダプタ（基底クラス）.

各種データソース（CSV/JSON/Parquet/既存DBなど）を共通インターフェースで扱うための
抽象基底クラスを定義します。
"""

from abc import ABC, abstractmethod

import polars as pl

# 標準列名への正規化（アダプタで揺れがある場合はここに寄せる）
STANDARD_COLUMNS = {
    "source_tag": str,  # 入力タグ（正規化前）
    "tag": str,  # 互換用（tag列があれば source_tag として扱う）
    "type_id": int | None,
    "format_id": int | None,
    "count": int | None,
    "deprecated_tags": str | None,
    "japanese": str | None,
    "translation": str | None,
}


class BaseAdapter(ABC):
    """入力ソースアダプタの基底クラス.

    全てのデータソースアダプタはこのクラスを継承し、
    read()/validate()/repair() を実装します。
    """

    @abstractmethod
    def read(self) -> pl.DataFrame | dict[str, pl.DataFrame]:
        """データソースを読み込み、Polars DataFrame（または辞書）に変換する.

        Returns:
            - 通常: 正規化された Polars DataFrame（STANDARD_COLUMNS 準拠を目標）
            - 複数テーブルソース（例: tags_v4.db）: DataFrame 辞書

        Raises:
            FileNotFoundError: ファイルが存在しない場合
            ValueError: データ形式が不正な場合
        """
        ...

    @abstractmethod
    def validate(self, df: pl.DataFrame) -> bool:
        """データ整合性を検証する."""
        ...

    @abstractmethod
    def repair(self, df: pl.DataFrame) -> pl.DataFrame:
        """壊れたデータを修復する（必要なら）."""
        ...
