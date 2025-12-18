from __future__ import annotations

import sqlite3
from pathlib import Path

import polars as pl

from genai_tag_db_dataset_builder import builder
from genai_tag_db_dataset_builder.core.database import create_database
from genai_tag_db_dataset_builder.core.master_data import initialize_master_data


def test_export_danbooru_view_parquet_creates_expected_columns(tmp_path: Path) -> None:
    db_path = tmp_path / "test.sqlite"
    create_database(db_path)
    initialize_master_data(db_path)

    conn = sqlite3.connect(db_path)
    try:
        # TAGS
        conn.executemany(
            "INSERT INTO TAGS (tag_id, source_tag, tag) VALUES (?, ?, ?)",
            [
                (1, "cat_ears", "cat ears"),
                (2, "nekomimi", "nekomimi"),
            ],
        )
        # TAG_STATUS (format_id=1)
        conn.executemany(
            "INSERT INTO TAG_STATUS (tag_id, format_id, type_id, alias, preferred_tag_id) VALUES (?, ?, ?, ?, ?)",
            [
                (1, 1, 0, 0, 1),  # canonical
                (2, 1, 0, 1, 1),  # alias -> preferred=1
            ],
        )
        # TAG_USAGE_COUNTS
        conn.execute(
            "INSERT INTO TAG_USAGE_COUNTS (tag_id, format_id, count) VALUES (?, ?, ?)",
            (1, 1, 123),
        )
        # TAG_TRANSLATIONS
        conn.executemany(
            "INSERT INTO TAG_TRANSLATIONS (translation_id, tag_id, language, translation) VALUES (?, ?, ?, ?)",
            [
                (1, 1, "ja", "猫耳"),
                (2, 1, "ja", "ねこみみ"),
                (3, 1, "zh", "猫耳"),
            ],
        )
        conn.commit()
    finally:
        conn.close()

    out_dir = tmp_path / "parquet"
    files = builder._export_danbooru_view_parquet(db_path, out_dir, chunk_size=10)
    assert files

    df = pl.read_parquet(files[0])
    assert df.columns == [
        "tag_id",
        "tag",
        "format_name",
        "type_name",
        "count",
        "deprecated_tags",
        "lang_ja",
        "lang_zh",
    ]

    row = df.to_dicts()[0]
    assert row["tag_id"] == 1
    assert row["tag"] == "cat ears"
    assert row["count"] == 123
    assert row["deprecated_tags"] == ["nekomimi"]
    assert row["lang_ja"] == ["猫耳", "ねこみみ"]
    assert row["lang_zh"] == ["猫耳"]
