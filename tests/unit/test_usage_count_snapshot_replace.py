import sqlite3
from pathlib import Path

from genai_tag_db_dataset_builder import builder
from genai_tag_db_dataset_builder.core.database import create_database


def test_replace_usage_counts_for_format_replaces_only_target_format(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    create_database(db_path)

    conn = sqlite3.connect(db_path)
    try:
        # seed: format_id=1 and format_id=2
        conn.executemany(
            "INSERT INTO TAG_USAGE_COUNTS (tag_id, format_id, count, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            [
                (1, 1, 10, "2000-01-01 00:00:00+00:00", "2000-01-01 00:00:00+00:00"),
                (2, 1, 20, "2000-01-01 00:00:00+00:00", "2000-01-01 00:00:00+00:00"),
                (1, 2, 99, "2001-01-01 00:00:00+00:00", "2001-01-01 00:00:00+00:00"),
            ],
        )
        conn.commit()

        ts = "2024-10-16 00:00:00+00:00"
        builder._replace_usage_counts_for_format(
            conn,
            format_id=1,
            counts_by_tag_id={1: 100, 3: 300},
            timestamp=ts,
        )

        rows_fmt1 = conn.execute(
            "SELECT tag_id, format_id, count, created_at, updated_at "
            "FROM TAG_USAGE_COUNTS WHERE format_id = 1 ORDER BY tag_id"
        ).fetchall()
        assert rows_fmt1 == [
            (1, 1, 100, ts, ts),
            (3, 1, 300, ts, ts),
        ]

        rows_fmt2 = conn.execute(
            "SELECT tag_id, format_id, count, created_at, updated_at "
            "FROM TAG_USAGE_COUNTS WHERE format_id = 2 ORDER BY tag_id"
        ).fetchall()
        assert rows_fmt2 == [
            (1, 2, 99, "2001-01-01 00:00:00+00:00", "2001-01-01 00:00:00+00:00"),
        ]
    finally:
        conn.close()

