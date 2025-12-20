"""tags_v4.db 取り込みアダプタ.

既存の tags_v4.db（genai-tag-db-tools）から必要テーブルを読み込み、
データセット構築用の Polars DataFrame 群として返します。
"""

from pathlib import Path

import polars as pl
from loguru import logger
from sqlalchemy import create_engine, text

from genai_tag_db_dataset_builder.core.normalize import canonicalize_source_tag

from .base_adapter import BaseAdapter


class Tags_v4_Adapter(BaseAdapter):
    """tags_v4.db 送出（エクスポート）アダプタ.

    genai-tag-db-tools の tags_v4.db から全テーブルを Polars DataFrame に変換します。

    Args:
        db_path: tags_v4.db のパス
    """

    def __init__(self, db_path: Path | str) -> None:
        """アダプタ初期化.

        Args:
            db_path: tags_v4.db のパス

        Raises:
            FileNotFoundError: データベースファイルが存在しない場合
        """
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")

        self.engine = create_engine(f"sqlite:///{self.db_path}")

    def read(self) -> dict[str, pl.DataFrame]:
        """全テーブルを読み込む.

        Returns:
            テーブル名をキーとする DataFrame の辞書
            - "tags": TAGS
            - "tag_status": TAG_STATUS
            - "tag_translations": TAG_TRANSLATIONS
            - "tag_usage_counts": TAG_USAGE_COUNTS

        Raises:
            ValueError: 読み込みに失敗した場合
        """
        tables: dict[str, pl.DataFrame] = {}

        # TAGSテーブル読み込み
        tags_query = "SELECT tag_id, tag, source_tag, created_at, updated_at FROM TAGS"
        tables["tags"] = pl.read_database(
            tags_query,
            connection=self.engine,
            schema_overrides={"created_at": pl.String, "updated_at": pl.String},
        )
        tables["tags"] = tables["tags"].with_columns(
            pl.col("source_tag")
            .map_elements(canonicalize_source_tag, return_dtype=pl.String)
            .alias("source_tag")
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

        # tags_v4.db は UNIQUE(tag) を持たないため、同一 tag の tag_id が複数存在し得る。
        # 新DB側では UNIQUE(tag) を採用するので、tag_id を代表（最小tag_id）へ付け替えてから dedup する。
        tags_df = tables["tags"]
        canonical = tags_df.group_by("tag").agg(pl.col("tag_id").min().alias("canonical_tag_id"))
        tags_with_canonical = tags_df.join(canonical, on="tag", how="left")
        remap_rows = tags_with_canonical.filter(pl.col("tag_id") != pl.col("canonical_tag_id")).select(
            [pl.col("tag_id"), pl.col("canonical_tag_id")]
        )

        if len(remap_rows) > 0:
            remap: dict[int, int] = dict(
                zip(
                    remap_rows["tag_id"].to_list(),
                    remap_rows["canonical_tag_id"].to_list(),
                    strict=False,
                )
            )

            def _remap_id(v: int) -> int:
                return remap.get(v, v)

            tables["tag_status"] = tables["tag_status"].with_columns(
                [
                    pl.col("tag_id").map_elements(_remap_id, return_dtype=pl.Int64).alias("tag_id"),
                    pl.col("preferred_tag_id")
                    .map_elements(_remap_id, return_dtype=pl.Int64)
                    .alias("preferred_tag_id"),
                ]
            )
            tables["tag_translations"] = tables["tag_translations"].with_columns(
                pl.col("tag_id").map_elements(_remap_id, return_dtype=pl.Int64).alias("tag_id")
            )
            tables["tag_usage_counts"] = tables["tag_usage_counts"].with_columns(
                pl.col("tag_id").map_elements(_remap_id, return_dtype=pl.Int64).alias("tag_id")
            )

        # TAGS の重複を統合（同一 tag、異なる tag_id を 1 つに寄せる）
        original_count = len(tables["tags"])
        tables["tags"] = self._deduplicate_tags(tables["tags"])
        deduplicated_count = len(tables["tags"])

        if original_count != deduplicated_count:
            logger.warning(
                f"Deduplicated {original_count - deduplicated_count} duplicate tags "
                f"from tags_v4.db (UNIQUE constraint enforcement)"
            )

        # TAG_STATUS の衝突検出（統合時の (tag, format_id) 競合など）
        conflicts = self._detect_tag_status_conflicts(tables["tag_status"], tables["tags"])
        if len(conflicts) > 0:
            logger.warning(f"Found {len(conflicts)} TAG_STATUS conflicts requiring review")
            # TODO: Export to CSV

        return tables

    def _deduplicate_tags(self, tags_df: pl.DataFrame) -> pl.DataFrame:
        """重複タグ（同一tag、異なるtag_id）を統合する.

        方針:
        - tag 列でグループ化
        - 最小 tag_id を残す（可能な限り既存IDを保持）
        - source_tag は「代表表記」として小文字に統一し、先頭のものを残す

        Args:
            tags_df: TAGSテーブルのDataFrame

        Returns:
            重複除去された DataFrame
        """
        tags_df = tags_df.with_columns(
            pl.col("source_tag")
            .map_elements(canonicalize_source_tag, return_dtype=pl.String)
            .alias("source_tag")
        )
        return tags_df.sort("tag_id").unique(subset=["tag"], keep="first")

    def _detect_tag_status_conflicts(
        self, tag_status_df: pl.DataFrame, tags_df: pl.DataFrame
    ) -> pl.DataFrame:
        """重複タグ統合時に TAG_STATUS の衝突を検出する.

        手動レビューが必要な衝突の一覧を DataFrame として返します。

        Args:
            tag_status_df: TAG_STATUSテーブルのDataFrame
            tags_df: 重複除去後のTAGSテーブルDataFrame

        Returns:
            衝突レコードの DataFrame
        """
        # TAG_STATUS に tag 列を付与（tag_id → tag）
        merged = tag_status_df.join(tags_df.select(["tag_id", "tag"]), on="tag_id", how="left")

        # tag が欠損している行（tags_v4.db 側の不整合など）で group key が NULL になると、
        # 無関係な複数タグが 1 グループに潰れて「衝突」に見えてしまう。
        # その場合は tag_id をキーにして、実際に同一 tag_id で矛盾があるかだけを見る。
        merged = merged.with_columns(
            [
                (pl.col("tag").is_null() | (pl.col("tag") == "")).alias("tag_missing"),
                pl.when(pl.col("tag").is_null() | (pl.col("tag") == ""))
                .then(pl.format("__missing_tag_id_{}__", pl.col("tag_id")))
                .otherwise(pl.col("tag"))
                .alias("tag_key"),
            ]
        )

        # (tag, format_id) 単位で集計し、衝突（複数バリアント）を検出
        conflicts = (
            merged.group_by(["tag_key", "format_id"])
            .agg(
                [
                    pl.col("alias").n_unique().alias("alias_variants"),
                    pl.col("type_id").n_unique().alias("type_id_variants"),
                    pl.col("preferred_tag_id").n_unique().alias("preferred_tag_id_variants"),
                ]
            )
            .filter(
                (pl.col("alias_variants") > 1)
                | (pl.col("type_id_variants") > 1)
                | (pl.col("preferred_tag_id_variants") > 1)
            )
        )

        return conflicts

    def validate(self, _df: pl.DataFrame) -> bool:
        """データ整合性検証（tags_v4.db は既存DBなので基本 True）."""
        return True

    def repair(self, df: pl.DataFrame) -> pl.DataFrame:
        """データ修復（tags_v4.db は既存DBなので基本不要）."""
        return df

    def get_existing_tags(self) -> set[str]:
        """既存 tag の集合を取得する（merge_tags用）."""
        query = "SELECT tag FROM TAGS"
        tags_df = pl.read_database(query, connection=self.engine)
        return set(tags_df["tag"].to_list())

    def get_next_tag_id(self) -> int:
        """次の tag_id（max(tag_id)+1）を取得する（merge_tags用）."""
        with self.engine.connect() as conn:
            result = conn.execute(text("SELECT MAX(tag_id) + 1 FROM TAGS"))
            next_id = result.scalar()
            return next_id if next_id is not None else 1
