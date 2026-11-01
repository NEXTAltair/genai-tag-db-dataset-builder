"""スキーマ版数の記録と、ベースDBから消えたタグの検出のテスト."""

import sqlite3
from pathlib import Path

from genai_tag_db_dataset_builder.builder import _find_removed_tags, build_dataset
from genai_tag_db_dataset_builder.core.database import SCHEMA_VERSION


def test_build_records_schema_version(tmp_path: Path) -> None:
    output = tmp_path / "out.sqlite"

    build_dataset(output, tmp_path, "v1", enable_site_tags=False)

    conn = sqlite3.connect(output)
    assert dict(conn.execute("SELECT key, value FROM DATABASE_METADATA")) == {
        "version": "v1",
        "schema_version": str(SCHEMA_VERSION),
    }


def test_find_removed_tags() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE TAGS (tag TEXT)")
    conn.executemany("INSERT INTO TAGS VALUES (?)", [("kept",), ("new",)])

    assert _find_removed_tags(conn, {"kept", "z removed", "a removed"}) == ["a removed", "z removed"]
    assert _find_removed_tags(conn, set()) == []


def test_build_writes_empty_removed_tags_report(tmp_path: Path) -> None:
    base = tmp_path / "base.sqlite"
    build_dataset(base, tmp_path, "v1", enable_site_tags=False)
    conn = sqlite3.connect(base)
    conn.execute("INSERT INTO TAGS (source_tag, tag) VALUES ('kept', 'kept')")
    conn.commit()
    conn.close()

    report_dir = tmp_path / "report"
    build_dataset(
        tmp_path / "out.sqlite",
        tmp_path,
        "v2",
        base_db_path=base,
        report_dir=report_dir,
        enable_site_tags=False,
    )

    assert (report_dir / "removed_tags.tsv").read_text(encoding="utf-8").splitlines() == ["tag"]
