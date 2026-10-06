from __future__ import annotations

import sqlite3
from pathlib import Path

from genai_tag_db_dataset_builder.builder import build_dataset


def _create_site_tags_sqlite(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE tags (
                id INTEGER PRIMARY KEY,
                tag TEXT NOT NULL,
                num INTEGER NOT NULL,
                type INTEGER NOT NULL,
                alias INTEGER,
                tag_jp TEXT
            );
            """
        )
        conn.execute(
            "INSERT INTO tags (id, tag, num, type, alias, tag_jp) VALUES (?, ?, ?, ?, ?, ?)",
            (1, "witch", 10, 0, None, "魔女､まじょ"),
        )
        conn.execute(
            "INSERT INTO tags (id, tag, num, type, alias, tag_jp) VALUES (?, ?, ?, ?, ?, ?)",
            (2, "sorceress", 5, 0, 1, None),
        )
        conn.commit()
    finally:
        conn.close()


def test_builder_imports_site_tags(tmp_path: Path) -> None:
    sources_dir = tmp_path / "sources"
    site_tags_root = sources_dir / "external_sources" / "site_tags" / "anime-pictures.net"
    sqlite_path = site_tags_root / "tags.sqlite"
    _create_site_tags_sqlite(sqlite_path)

    output_db = tmp_path / "out.db"
    report_dir = tmp_path / "reports"

    build_dataset(
        output_path=output_db,
        sources_dir=sources_dir,
        version="test",
        report_dir=report_dir,
        overwrite=True,
    )

    conn = sqlite3.connect(output_db)
    try:
        tags = {row[0] for row in conn.execute("SELECT tag FROM TAGS").fetchall()}
        assert "witch" in tags
        assert "sorceress" in tags

        witch_id = conn.execute("SELECT tag_id FROM TAGS WHERE tag = 'witch'").fetchone()[0]
        sorceress_id = conn.execute("SELECT tag_id FROM TAGS WHERE tag = 'sorceress'").fetchone()[0]

        status = conn.execute(
            "SELECT alias, preferred_tag_id FROM TAG_STATUS WHERE tag_id = ?",
            (sorceress_id,),
        ).fetchone()
        assert status == (1, witch_id)

        translations = conn.execute(
            "SELECT translation FROM TAG_TRANSLATIONS WHERE tag_id = ? AND language = 'ja'",
            (witch_id,),
        ).fetchall()
        assert sorted(row[0] for row in translations) == ["まじょ", "魔女"]
    finally:
        conn.close()


def test_builder_skips_site_tags_when_disabled(tmp_path: Path) -> None:
    """external_sources/site_tags が残っていても enable_site_tags=False なら取り込まない (CC0/MIT 混入防止)."""
    sources_dir = tmp_path / "sources"
    sqlite_path = sources_dir / "external_sources" / "site_tags" / "anime-pictures.net" / "tags.sqlite"
    _create_site_tags_sqlite(sqlite_path)

    output_db = tmp_path / "out.db"

    build_dataset(
        output_path=output_db,
        sources_dir=sources_dir,
        version="test",
        report_dir=tmp_path / "reports",
        overwrite=True,
        enable_site_tags=False,
    )

    conn = sqlite3.connect(output_db)
    try:
        tags = {row[0] for row in conn.execute("SELECT tag FROM TAGS").fetchall()}
        assert "witch" not in tags
        assert "sorceress" not in tags
    finally:
        conn.close()
