"""Tags_v4_Adapter for exporting tags_v4.db to Polars DataFrame.

This adapter exports all data from the existing tags_v4.db database
(genai-tag-db-tools) into Polars DataFrames for dataset building.
"""

from pathlib import Path

import polars as pl
from sqlalchemy import create_engine, text

from .base_adapter import BaseAdapter


class Tags_v4_Adapter(BaseAdapter):
    """tags_v4.db逆エクスポートアダプタ.

    genai-tag-db-toolsのtags_v4.dbから全テーブルをPolars DataFrameに変換します。

    Args:
        db_path: tags_v4.dbのパス
    """

    def __init__(self, db_path: Path | str) -> None:
        """Initialize adapter.

        Args:
            db_path: Path to tags_v4.db

        Raises:
            FileNotFoundError: データベースファイルが存在しない場合
        """
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")

        self.engine = create_engine(f"sqlite:///{self.db_path}")

    def read(self) -> dict[str, pl.DataFrame]:
        """全テーブルを読み込み.

        Returns:
            テーブル名をキーとするDataFrameの辞書
            - "tags": TAGSテーブル
            - "tag_status": TAG_STATUSテーブル
            - "tag_translations": TAG_TRANSLATIONSテーブル
            - "tag_usage_counts": TAG_USAGE_COUNTSテーブル

        Raises:
            ValueError: データ読み込みに失敗した場合
        """
        tables = {}

        # TAGSテーブル読み込み
        tags_query = "SELECT tag_id, tag, source_tag, created_at, updated_at FROM TAGS"
        tables["tags"] = pl.read_database(
            tags_query,
            connection=self.engine,
            schema_overrides={"created_at": pl.String, "updated_at": pl.String},
        )

        # TAG_STATUSテーブル読み込み
        status_query = """
            SELECT tag_id, format_id, type_id, alias, preferred_tag_id,
                   created_at, updated_at
            FROM TAG_STATUS
        """
        tables["tag_status"] = pl.read_database(
            status_query,
            connection=self.engine,
            schema_overrides={"created_at": pl.String, "updated_at": pl.String},
        )

        # TAG_TRANSLATIONSテーブル読み込み
        translations_query = """
            SELECT translation_id, tag_id, language, translation,
                   created_at, updated_at
            FROM TAG_TRANSLATIONS
        """
        tables["tag_translations"] = pl.read_database(
            translations_query,
            connection=self.engine,
            schema_overrides={"created_at": pl.String, "updated_at": pl.String},
        )

        # TAG_USAGE_COUNTSテーブル読み込み
        counts_query = """
            SELECT tag_id, format_id, count, created_at, updated_at
            FROM TAG_USAGE_COUNTS
        """
        tables["tag_usage_counts"] = pl.read_database(
            counts_query,
            connection=self.engine,
            schema_overrides={"created_at": pl.String, "updated_at": pl.String},
        )

        return tables

    def validate(self, df: pl.DataFrame) -> bool:
        """データ整合性検証.

        Args:
            df: 検証対象のDataFrame

        Returns:
            検証成功の場合True

        Note:
            tags_v4.dbは既に整合性が保たれているため、基本的にTrueを返す
        """
        return True

    def repair(self, df: pl.DataFrame) -> pl.DataFrame:
        """データ修復.

        Args:
            df: 修復対象のDataFrame

        Returns:
            修復されたDataFrame（tags_v4.dbは修復不要なのでそのまま返却）
        """
        return df

    def get_existing_tags(self) -> set[str]:
        """既存タグのset取得（merge_tags用）.

        Returns:
            既存のtag列のset
        """
        query = "SELECT tag FROM TAGS"
        tags_df = pl.read_database(query, connection=self.engine)
        return set(tags_df["tag"].to_list())

    def get_next_tag_id(self) -> int:
        """次のtag_id取得（merge_tags用）.

        Returns:
            max(tag_id) + 1
        """
        with self.engine.connect() as conn:
            result = conn.execute(text("SELECT MAX(tag_id) + 1 FROM TAGS"))
            next_id = result.scalar()
            return next_id if next_id is not None else 1
