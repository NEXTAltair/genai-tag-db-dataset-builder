"""Tag merging and conflict detection.

This module provides tag merging functionality using set difference approach
and conflict detection for type_id mismatches and alias changes.
"""

import polars as pl

from .normalize import normalize_tag


def merge_tags(
    existing_tags: set[str],
    new_df: pl.DataFrame,
    next_tag_id: int,
) -> pl.DataFrame:
    """新規タグをマージ（set差分方式）.

    既存のtag一覧と新規DataFrameを比較し、新規タグのみを抽出してtag_idを採番します。

    Args:
        existing_tags: 既存のtag set（tags_v4.dbから取得）
        new_df: 新規ソース（source_tag列を含む）
        next_tag_id: 次に割り当てるtag_id（max(tag_id)+1）

    Returns:
        追加するタグのDataFrame（tag_id, tag, source_tag列）

    Examples:
        >>> existing_tags = {"witch", "spiked collar"}
        >>> new_df = pl.DataFrame({"source_tag": ["Witch", "new_tag"]})
        >>> result = merge_tags(existing_tags, new_df, 100)
        >>> result.columns
        ['tag_id', 'tag', 'source_tag']
    """
    # 1. 新規source_tagを正規化してtag列を生成
    new_df = new_df.with_columns(pl.col("source_tag").map_elements(normalize_tag, return_dtype=pl.String).alias("tag"))

    # 2. 既存tagとの差分抽出（set差分）
    new_tags_df = new_df.filter(~pl.col("tag").is_in(existing_tags))

    # 3. 重複除去（同じtagが複数source_tagから来る場合）
    new_tags_df = new_tags_df.unique(subset=["tag"])

    # 4. 再現性のためtagでソート
    new_tags_df = new_tags_df.sort("tag")

    # 5. tag_id採番（max+1から連番）
    new_tags_df = new_tags_df.with_row_index("row_num").with_columns(
        (pl.col("row_num") + next_tag_id).cast(pl.Int64).alias("tag_id")
    ).drop("row_num")

    # 6. 必要な列のみ返却
    return new_tags_df.select(["tag_id", "tag", "source_tag"])


def process_deprecated_tags(
    canonical_tag: str,
    deprecated_tags: str,
    format_id: int,
    tags_mapping: dict[str, int],
) -> list[dict[str, int]]:
    """deprecated_tags列からTAG_STATUSレコード生成.

    canonical_tag自身のレコード（alias=0）と、deprecated_tagsから抽出した
    aliasレコード（alias=1）を生成します。

    Args:
        canonical_tag: 正規タグ（エイリアスの参照先）
        deprecated_tags: カンマ区切りのdeprecated tags文字列
        format_id: フォーマットID
        tags_mapping: tag → tag_id のマッピング辞書

    Returns:
        TAG_STATUSレコードのリスト

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
    records = []

    # canonical自身（alias=0）
    records.append(
        {
            "tag_id": canonical_tag_id,
            "format_id": format_id,
            "alias": 0,
            "preferred_tag_id": canonical_tag_id,
        }
    )

    # aliasレコード（alias=1）
    if deprecated_tags:
        for alias_source_tag in deprecated_tags.split(","):
            alias_tag = normalize_tag(alias_source_tag.strip())
            if alias_tag in tags_mapping:
                alias_tag_id = tags_mapping[alias_tag]
                records.append(
                    {
                        "tag_id": alias_tag_id,
                        "format_id": format_id,
                        "alias": 1,
                        "preferred_tag_id": canonical_tag_id,
                    }
                )
            # Note: deprecated_tags に含まれる alias が tags_mapping に存在しない場合、
            # その alias は TAGS テーブルに存在しないため、TAG_STATUS レコードを作成できない。
            # 事前に deprecated_tags 内の全タグを TAGS に投入する処理が必要。

    return records


def detect_conflicts(
    existing_df: pl.DataFrame,
    new_df: pl.DataFrame,
    tags_mapping: dict[str, int] | None = None,
) -> dict[str, pl.DataFrame]:
    """tag + format_id でJOIN（tag_idは後から引く）.

    既存TAG_STATUSと新規データを比較し、type_id不一致とalias変更を検出します。

    Args:
        existing_df: 既存TAG_STATUS（tag列JOIN済み）
        new_df: 新規データ（tag列あり）
        tags_mapping: tag → tag_id のマッピング辞書（現在未使用、将来の拡張用）

    Returns:
        衝突情報の辞書
        - "type_conflicts": type_id不一致のDataFrame
        - "alias_changes": alias変更のDataFrame

    Examples:
        >>> existing = pl.DataFrame({
        ...     "tag": ["witch"],
        ...     "format_id": [1],
        ...     "type_id": [4],
        ...     "alias": [0]
        ... })
        >>> new = pl.DataFrame({
        ...     "tag": ["witch"],
        ...     "format_id": [1],
        ...     "type_id": [0],
        ...     "alias": [1]
        ... })
        >>> conflicts = detect_conflicts(existing, new)
        >>> "type_conflicts" in conflicts
        True
    """
    # tags_mapping は現在未使用だが、将来の拡張用に引数として保持
    _ = tags_mapping

    merged = existing_df.join(new_df, on=["tag", "format_id"], how="inner", suffix="_new")

    # type_id不一致
    type_conflicts = merged.filter(pl.col("type_id") != pl.col("type_id_new"))

    # alias変更（既存=0、新規=1）
    alias_changes = merged.filter((pl.col("alias") == 0) & (pl.col("alias_new") == 1))

    return {
        "type_conflicts": type_conflicts,
        "alias_changes": alias_changes,
    }
