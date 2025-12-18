from __future__ import annotations

import sqlite3
from pathlib import Path

import polars as pl

from genai_tag_db_dataset_builder.adapters.site_tags_adapter import SiteTagsAdapter


def _collect_all_chunks(adapter: SiteTagsAdapter) -> pl.DataFrame:
    dfs: list[pl.DataFrame] = []
    for c in adapter.iter_chunks(chunk_size=10_000):
        dfs.append(c.df)
    return pl.concat(dfs) if dfs else pl.DataFrame()


def test_site_tags_adapter_danbooru_aliases_become_deprecated_tags(tmp_path: Path) -> None:
    db = tmp_path / "tags.sqlite"
    conn = sqlite3.connect(db)
    try:
        conn.executescript(
            """
            CREATE TABLE tags (
              "index" INTEGER,
              "id" INTEGER,
              "name" TEXT,
              "post_count" INTEGER,
              "category" INTEGER,
              "created_at" TIMESTAMP,
              "updated_at" TIMESTAMP,
              "is_deprecated" INTEGER,
              "words" TEXT
            );
            CREATE TABLE tag_aliases (
              "index" INTEGER,
              "alias" TEXT,
              "tag" TEXT
            );
            """
        )
        conn.execute(
            "INSERT INTO tags (id, name, post_count, category, is_deprecated, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (1, "cat_girl", 10, 0, 0, "2020-01-01 00:00:00", "2024-01-01 00:00:00"),
        )
        # deprecated source tag (alias)
        conn.execute(
            "INSERT INTO tags (id, name, post_count, category, is_deprecated) VALUES (?, ?, ?, ?, ?)",
            (2, "neko_girl", 1, 0, 1),
        )
        conn.execute(
            "INSERT INTO tag_aliases (alias, tag) VALUES (?, ?)",
            ("neko_girl", "cat_girl"),
        )
        conn.commit()
    finally:
        conn.close()

    adapter = SiteTagsAdapter(db, format_id=1)
    df = _collect_all_chunks(adapter)

    assert "source_tag" in df.columns
    # alias側は行としては出ず、推奨側の deprecated_tags に入る
    got = {r["source_tag"]: r.get("deprecated_tags", "") for r in df.to_dicts()}
    assert "cat_girl" in got
    assert got["cat_girl"] == "neko_girl"
    assert "neko_girl" not in got


def test_site_tags_adapter_e621_invalid_sink_becomes_deprecated_tag_row(tmp_path: Path) -> None:
    db = tmp_path / "tags.sqlite"
    conn = sqlite3.connect(db)
    try:
        conn.executescript(
            """
            CREATE TABLE tags (
              "index" INTEGER,
              "id" INTEGER,
              "name" TEXT,
              "post_count" INTEGER,
              "related_tags" TEXT,
              "related_tags_updated_at" TEXT,
              "category" INTEGER,
              "is_locked" INTEGER,
              "created_at" TIMESTAMP,
              "updated_at" TIMESTAMP
            );
            CREATE TABLE tag_aliases (
              "index" INTEGER,
              "alias" TEXT,
              "tag" TEXT
            );
            """
        )
        conn.execute(
            "INSERT INTO tags (id, name, post_count, category) VALUES (?, ?, ?, ?)",
            (10, "absurd_res", 123, 7),
        )
        # invalid_tag への吸い込み（redirectしないため、deprecated なタグとして独立投入される）
        conn.execute(
            "INSERT INTO tag_aliases (alias, tag) VALUES (?, ?)",
            (")", "invalid_tag"),
        )
        conn.commit()
    finally:
        conn.close()

    adapter = SiteTagsAdapter(db, format_id=2)
    df = _collect_all_chunks(adapter)

    rows = df.to_dicts()
    assert any(r["source_tag"] == "absurd_res" for r in rows)
    invalid_rows = [r for r in rows if r["source_tag"] == ")"]
    assert invalid_rows, "invalid_tag sink alias should be emitted as a deprecated row"
    assert invalid_rows[0]["deprecated"] == 1
    assert invalid_rows[0]["type_id"] == 6  # e621 invalid category


def test_site_tags_adapter_sankaku_sparse_translation_columns_are_not_dropped(
    tmp_path: Path,
) -> None:
    """低頻度の翻訳列（例: trans_zh-CN）が chunk 内に存在しても落ちないこと."""
    db = tmp_path / "tags.sqlite"
    conn = sqlite3.connect(db)
    try:
        conn.executescript(
            """
            CREATE TABLE tags (
              "index" INTEGER,
              "id" INTEGER,
              "name" TEXT,
              "type" INTEGER,
              "post_count" INTEGER,
              "pool_count" INTEGER,
              "series_count" INTEGER,
              "trans_en" TEXT,
              "trans_zh-CN" TEXT
            );
            """
        )
        # 先頭100行には trans_zh-CN を入れない（Polarsの既定推定長で落ちやすい）
        for i in range(150):
            conn.execute(
                "INSERT INTO tags (id, name, type, post_count, pool_count, series_count, trans_en, \"trans_zh-CN\") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (i, f"tag_{i}", 0, 1, 0, 0, f"en_{i}", None),
            )
        # 後半にのみ trans_zh-CN が入る
        conn.execute(
            "INSERT INTO tags (id, name, type, post_count, pool_count, series_count, trans_en, \"trans_zh-CN\") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (999, "rape", 0, 0, 0, 0, "rape", "強姦"),
        )
        conn.commit()
    finally:
        conn.close()

    adapter = SiteTagsAdapter(db, format_id=6)
    df = _collect_all_chunks(adapter)

    assert "zh-CN" in df.columns
    row = df.filter(pl.col("source_tag") == "rape")
    assert row.height == 1
    assert row["zh-CN"][0] == "強姦"


def test_site_tags_adapter_negative_count_is_treated_as_missing(tmp_path: Path) -> None:
    db = tmp_path / "tags.sqlite"
    conn = sqlite3.connect(db)
    try:
        conn.executescript(
            """
            CREATE TABLE tags (
              "name" TEXT,
              "post_count" INTEGER,
              "category" INTEGER
            );
            """
        )
        conn.execute(
            "INSERT INTO tags (name, post_count, category) VALUES (?, ?, ?)",
            ("neg_count", -10, 0),
        )
        conn.commit()
    finally:
        conn.close()

    adapter = SiteTagsAdapter(db, format_id=2)
    df = _collect_all_chunks(adapter)

    row = df.filter(pl.col("source_tag") == "neg_count")
    assert row.height == 1
    assert row["count"][0] is None
