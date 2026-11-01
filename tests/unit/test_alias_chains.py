"""alias の連鎖解消と健全性レポートのテスト."""

import sqlite3
from pathlib import Path

from genai_tag_db_dataset_builder.builder import _flatten_alias_chains
from genai_tag_db_dataset_builder.core.database import create_schema
from genai_tag_db_dataset_builder.tools.report_db_health import run_health_checks


def _status_db(path: Path, rows: list[tuple[int, int, int, int]]) -> sqlite3.Connection:
    sqlite3.connect(path).close()
    create_schema(path)
    conn = sqlite3.connect(path)
    tag_ids = {row[0] for row in rows} | {row[3] for row in rows}
    conn.executemany(
        "INSERT INTO TAGS (tag_id, source_tag, tag) VALUES (?, ?, ?)",
        [(tag_id, f"t{tag_id}", f"t{tag_id}") for tag_id in sorted(tag_ids)],
    )
    conn.executemany(
        "INSERT INTO TAG_STATUS (tag_id, format_id, type_id, alias, preferred_tag_id) VALUES (?, ?, 0, ?, ?)",
        rows,
    )
    conn.commit()
    return conn


def test_flatten_alias_chains(tmp_path: Path) -> None:
    conn = _status_db(
        tmp_path / "db.sqlite",
        [
            (1, 1, 1, 2),
            (2, 1, 1, 3),
            (3, 1, 1, 4),
            (4, 1, 0, 4),
            (5, 1, 1, 4),
            (2, 2, 1, 4),
            (4, 2, 0, 4),
            (1, 2, 1, 2),
        ],
    )

    assert _flatten_alias_chains(conn) == 3

    rows = {
        (tag_id, format_id): preferred
        for tag_id, format_id, preferred in conn.execute(
            "SELECT tag_id, format_id, preferred_tag_id FROM TAG_STATUS WHERE alias = 1"
        )
    }
    assert rows == {(1, 1): 4, (2, 1): 4, (3, 1): 4, (5, 1): 4, (2, 2): 4, (1, 2): 4}
    assert _flatten_alias_chains(conn) == 0


def test_flatten_alias_chains_leaves_cycles_and_missing_targets(tmp_path: Path) -> None:
    conn = _status_db(
        tmp_path / "db.sqlite",
        [
            (1, 1, 1, 2),
            (2, 1, 1, 1),
            (3, 1, 1, 4),
        ],
    )

    assert _flatten_alias_chains(conn) == 0
    assert conn.execute("SELECT tag_id, preferred_tag_id FROM TAG_STATUS ORDER BY tag_id").fetchall() == [
        (1, 2),
        (2, 1),
        (3, 4),
    ]


def test_health_report_counts_chains_and_missing_targets(tmp_path: Path) -> None:
    db_path = tmp_path / "db.sqlite"
    conn = _status_db(
        db_path,
        [
            (1, 1, 1, 2),
            (2, 1, 1, 3),
            (3, 1, 0, 3),
            (4, 1, 1, 5),
        ],
    )
    conn.close()

    summary = run_health_checks(db_path, tmp_path / "health").read_text(encoding="utf-8")

    assert "alias_chains\t1" in summary
    assert "alias_missing_preferred\t1" in summary
