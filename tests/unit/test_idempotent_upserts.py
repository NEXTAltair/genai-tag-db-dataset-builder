import sqlite3
from pathlib import Path

from genai_tag_db_dataset_builder.builder import (
    _TAG_STATUS_UPSERT_IF_CHANGED_SQL,
    _TAG_USAGE_COUNTS_UPSERT_IF_GREATER_SQL,
)
from genai_tag_db_dataset_builder.core.database import create_database
from genai_tag_db_dataset_builder.core.master_data import initialize_master_data


def test_tag_status_upsert_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "t.sqlite"
    create_database(db_path)
    initialize_master_data(db_path)

    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "INSERT INTO TAGS (tag_id, tag, source_tag) VALUES (?, ?, ?)",
            (1, "cat", "cat"),
        )
        conn.commit()

        before = conn.total_changes
        conn.execute(_TAG_STATUS_UPSERT_IF_CHANGED_SQL, (1, 1, 0, 0, 1))
        conn.commit()
        after_first = conn.total_changes

        conn.execute(_TAG_STATUS_UPSERT_IF_CHANGED_SQL, (1, 1, 0, 0, 1))
        conn.commit()
        after_second = conn.total_changes

        assert after_first > before
        assert after_second == after_first  # no-op update
    finally:
        conn.close()


def test_usage_counts_upsert_updates_only_if_greater(tmp_path: Path) -> None:
    db_path = tmp_path / "t.sqlite"
    create_database(db_path)
    initialize_master_data(db_path)

    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "INSERT INTO TAGS (tag_id, tag, source_tag) VALUES (?, ?, ?)",
            (1, "cat", "cat"),
        )
        conn.commit()

        before = conn.total_changes
        conn.execute(_TAG_USAGE_COUNTS_UPSERT_IF_GREATER_SQL, (1, 1, 10))
        conn.commit()
        after_first = conn.total_changes

        conn.execute(_TAG_USAGE_COUNTS_UPSERT_IF_GREATER_SQL, (1, 1, 10))
        conn.commit()
        after_equal = conn.total_changes

        conn.execute(_TAG_USAGE_COUNTS_UPSERT_IF_GREATER_SQL, (1, 1, 9))
        conn.commit()
        after_lower = conn.total_changes

        conn.execute(_TAG_USAGE_COUNTS_UPSERT_IF_GREATER_SQL, (1, 1, 11))
        conn.commit()
        after_higher = conn.total_changes

        assert after_first > before
        assert after_equal == after_first
        assert after_lower == after_first
        assert after_higher > after_first
    finally:
        conn.close()

