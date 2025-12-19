"""タグのマージと衝突検出.

- set差分方式による新規タグ抽出と採番
- deprecated_tags からの alias（TAG_STATUS）レコード生成
- type_id / alias 変更の衝突検出
"""

import polars as pl
from loguru import logger

from .normalize import canonicalize_source_tag, normalize_tag


def merge_tags(
    existing_tags: set[str],
    new_df: pl.DataFrame,
    next_tag_id: int,
) -> pl.DataFrame:
    """新規タグをマージする（set差分方式）.

    既存 tag 一覧と新規 DataFrame を比較し、新規タグだけを抽出して tag_id を採番します。

    Args:
        existing_tags: 既存 tag の集合（例: tags_v4.db から取得）
        new_df: 新規ソース（source_tag 列を含む必要あり）
        next_tag_id: 次に割り当てる tag_id（通常は max(tag_id)+1）

    Returns:
        追加するタグの DataFrame（tag_id, tag, source_tag）

    Raises:
        ValueError: new_df に 'source_tag' 列が存在しない場合

    Examples:
        >>> existing_tags = {"witch", "spiked collar"}
        >>> new_df = pl.DataFrame({"source_tag": ["Witch", "new_tag"]})
        >>> result = merge_tags(existing_tags, new_df, 100)
        >>> result.columns
        ['tag_id', 'tag', 'source_tag']
    """
    if "source_tag" not in new_df.columns:
        raise ValueError(
            "merge_tags() requires 'source_tag' column in new_df. "
            "Ensure adapter normalization completed successfully."
        )

    # 1) source_tag（代表表記）を小文字統一（顔文字はそのまま）
    new_df = new_df.with_columns(
        pl.col("source_tag")
        .map_elements(canonicalize_source_tag, return_dtype=pl.String)
        .alias("source_tag")
    )

    # 2) source_tag から tag（TAGS.tag）を生成
    new_df = new_df.with_columns(
        pl.col("source_tag").map_elements(normalize_tag, return_dtype=pl.String).alias("tag")
    )

    # 3) 既存 tag との差分を抽出
    new_tags_df = new_df.filter(~pl.col("tag").is_in(existing_tags))

    # 4) 重複除去（同じ tag が複数 source_tag から来る場合）
    new_tags_df = new_tags_df.unique(subset=["tag"])

    # 5) 再現性のため tag でソート
    new_tags_df = new_tags_df.sort("tag")

    # 6) tag_id 採番（next_tag_id からの連番）
    new_tags_df = (
        new_tags_df.with_row_index("row_num")
        .with_columns((pl.col("row_num") + next_tag_id).cast(pl.Int64).alias("tag_id"))
        .drop("row_num")
    )

    return new_tags_df.select(["tag_id", "tag", "source_tag"])


def process_deprecated_tags(
    canonical_tag: str,
    deprecated_tags: str,
    format_id: int,
    tags_mapping: dict[str, int],
) -> list[dict[str, int]]:
    """deprecated_tags列から TAG_STATUS レコードを生成する.

    canonical_tag 自身のレコード（alias=0）と、
    deprecated_tags から抽出した alias レコード（alias=1）を生成します。

    Preconditions:
        - deprecated_tags 内の全タグが tags_mapping に存在すること
        - この条件を満たすため、呼び出し側は2パス処理が必要
          (Pass 1: 全タグ収集・TAGS登録 / Pass 2: TAG_STATUS作成)

    Args:
        canonical_tag: 正規タグ（エイリアスの参照先）
        deprecated_tags: カンマ区切りの deprecated tags 文字列
        format_id: フォーマットID
        tags_mapping: tag → tag_id のマッピング辞書

    Returns:
        TAG_STATUS レコードのリスト（dict）

    Examples:
        >>> tags_mapping = {"witch": 1, "mage": 2}
        >>> records = process_deprecated_tags("witch", "mage", 1, tags_mapping)
        >>> len(records)
        2
        >>> records[0]["alias"]
        0
        >>> records[1]["alias"]
        1
    """
    canonical_tag_id = tags_mapping[canonical_tag]
    records: list[dict[str, int]] = []

    # canonical 自身（alias=0）
    records.append(
        {
            "tag_id": canonical_tag_id,
            "format_id": format_id,
            "alias": 0,
            "preferred_tag_id": canonical_tag_id,
        }
    )

    # alias レコード（alias=1）
    skipped_aliases: list[str] = []
    if deprecated_tags:
        for alias_source_tag in deprecated_tags.split(","):
            alias_tag = normalize_tag(alias_source_tag.strip())
            if alias_tag in tags_mapping:
                alias_tag_id = tags_mapping[alias_tag]
                if alias_tag_id == canonical_tag_id:
                    # deprecated_tags に canonical 自身が混ざっている場合がある。
                    # TAG_STATUS は (tag_id, format_id) が主キーなので、ここで上書きされると
                    # alias=1 & preferred_tag_id==tag_id となり CHECK 制約に違反するため除外する。
                    continue
                records.append(
                    {
                        "tag_id": alias_tag_id,
                        "format_id": format_id,
                        "alias": 1,
                        "preferred_tag_id": canonical_tag_id,
                    }
                )
            else:
                # tags_mapping に存在しない alias はスキップ
                skipped_aliases.append(alias_tag)

    # スキップされた alias があれば警告
    if skipped_aliases:
        logger.warning(
            f"Skipped {len(skipped_aliases)} alias(es) missing from TAGS table: "
            f"{', '.join(skipped_aliases)} (canonical: {canonical_tag}, format_id: {format_id}). "
            f"Ensure two-pass processing: collect all tags first, then create TAG_STATUS."
        )

    return records


def detect_conflicts(
    existing_df: pl.DataFrame,
    new_df: pl.DataFrame,
    tags_mapping: dict[str, int] | None = None,
) -> dict[str, pl.DataFrame]:
    """tag + format_id でJOINして衝突を検出する.

    既存 TAG_STATUS と新規データを比較し、type_id 不一致と alias 変更を検出します。

    Args:
        existing_df: 既存 TAG_STATUS（tag 列JOIN済み）
        new_df: 新規データ（tag 列あり）
        tags_mapping: tag → tag_id のマッピング辞書（現在未使用、将来拡張用）

    Returns:
        衝突情報の辞書
        - "type_conflicts": type_id不一致のDataFrame
        - "alias_changes": alias変更のDataFrame
    """
    # tags_mapping は現在未使用（将来拡張用に引数として保持）
    _ = tags_mapping

    merged = existing_df.join(new_df, on=["tag", "format_id"], how="inner", suffix="_new")

    type_conflicts = merged.filter(pl.col("type_id") != pl.col("type_id_new"))

    # alias変更（既存が canonical、新規が alias）
    alias_changes = merged.filter((pl.col("alias") == 0) & (pl.col("alias_new") == 1))

    return {
        "type_conflicts": type_conflicts,
        "alias_changes": alias_changes,
    }
