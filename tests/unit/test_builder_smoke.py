from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from genai_tag_db_dataset_builder.builder import build_dataset


def _create_minimal_tags_v4_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE TAGS (
                tag_id INTEGER NOT NULL PRIMARY KEY,
                tag TEXT NOT NULL,
                source_tag TEXT NOT NULL,
                created_at TEXT,
                updated_at TEXT
            );
            CREATE TABLE TAG_STATUS (
                tag_id INTEGER NOT NULL,
                format_id INTEGER NOT NULL,
                type_id INTEGER NOT NULL,
                alias INTEGER NOT NULL,
                preferred_tag_id INTEGER NOT NULL,
                created_at TEXT,
                updated_at TEXT
            );
            CREATE TABLE TAG_TRANSLATIONS (
                translation_id INTEGER NOT NULL PRIMARY KEY,
                tag_id INTEGER NOT NULL,
                language TEXT,
                translation TEXT,
                created_at TEXT,
                updated_at TEXT
            );
            CREATE TABLE TAG_USAGE_COUNTS (
                tag_id INTEGER NOT NULL,
                format_id INTEGER NOT NULL,
                count INTEGER NOT NULL,
                created_at TEXT,
                updated_at TEXT
            );
            """
        )

        # baseline: witch (danbooru/general)
        conn.execute(
            "INSERT INTO TAGS (tag_id, tag, source_tag, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (1, "witch", "witch", "2025-01-01", "2025-01-01"),
        )
        conn.execute(
            "INSERT INTO TAG_STATUS (tag_id, format_id, type_id, alias, preferred_tag_id, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (1, 1, 0, 0, 1, "2025-01-01", "2025-01-01"),
        )
        conn.execute(
            "INSERT INTO TAG_TRANSLATIONS (translation_id, tag_id, language, translation, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (1, 1, "ja", "魔女", "2025-01-01", "2025-01-01"),
        )
        conn.execute(
            "INSERT INTO TAG_USAGE_COUNTS (tag_id, format_id, count, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (1, 1, 10, "2025-01-01", "2025-01-01"),
        )
        conn.commit()
    finally:
        conn.close()


def test_builder_builds_minimal_db(tmp_path: Path) -> None:
    sources_dir = tmp_path / "sources"

    # tags_v4.db
    tags_v4_path = sources_dir / "tags_v4.db"
    _create_minimal_tags_v4_db(tags_v4_path)

    # CSV sources
    csv_root = sources_dir / "TagDB_DataSource_CSV" / "A"
    csv_root.mkdir(parents=True, exist_ok=True)

    # tag source: adds new tag and alias
    (csv_root / "danbooru.csv").write_text(
        "source_tag,type_id,format_id,deprecated_tags,count\nWitch,0,1,sorceress,11\nnew_tag,0,1,,1\n",
        encoding="utf-8",
    )

    # conflicting count (same tag/format) -> max is chosen
    (csv_root / "e621.csv").write_text(
        "source_tag,type_id,format_id,deprecated_tags,count\nwitch,0,2,,5\nwitch,0,2,,500\n",
        encoding="utf-8",
    )

    # translation source (ja)
    translation_file = csv_root / "danbooru_machine_jp.csv"
    translation_file.write_text(
        "tag,japanese\nwitch,魔女\nnew_tag,新しいタグ\n",
        encoding="utf-8",
    )

    # override to allow translation source (UNKNOWN判定回避)
    overrides_path = tmp_path / "column_type_overrides.json"
    overrides_path.write_text(
        json.dumps({str(translation_file): {"tag": "source"}}),
        encoding="utf-8",
    )

    output_db = tmp_path / "out.db"
    report_dir = tmp_path / "reports"

    build_dataset(
        output_path=output_db,
        sources_dir=sources_dir,
        version="test",
        report_dir=report_dir,
        overrides_path=overrides_path,
        overwrite=True,
    )

    conn = sqlite3.connect(output_db)
    try:
        tags = conn.execute("SELECT tag, source_tag FROM TAGS").fetchall()
        tags_by_tag = {t: s for (t, s) in tags}
        assert "witch" in tags_by_tag
        assert tags_by_tag["witch"] == "witch"
        assert "new tag" in tags_by_tag  # underscore -> space
        assert "sorceress" in tags_by_tag

        # translations (ja)
        rows = conn.execute(
            "SELECT t.tag, tr.language, tr.translation "
            "FROM TAG_TRANSLATIONS tr JOIN TAGS t ON t.tag_id = tr.tag_id "
            "WHERE tr.language = 'ja'"
        ).fetchall()
        assert ("witch", "ja", "魔女") in rows
        assert ("new tag", "ja", "新しいタグ") in rows

        # status: canonical record exists for new tag
        new_tag_id = conn.execute("SELECT tag_id FROM TAGS WHERE tag = 'new tag'").fetchone()[0]
        st = conn.execute(
            "SELECT alias, preferred_tag_id FROM TAG_STATUS WHERE tag_id = ? AND format_id = 1",
            (new_tag_id,),
        ).fetchone()
        assert st == (0, new_tag_id)

        # usage count: max is chosen for same (tag_id, format_id)
        witch_id = conn.execute("SELECT tag_id FROM TAGS WHERE tag = 'witch'").fetchone()[0]
        c = conn.execute(
            "SELECT count FROM TAG_USAGE_COUNTS WHERE tag_id = ? AND format_id = 2",
            (witch_id,),
        ).fetchone()[0]
        assert c == 500
    finally:
        conn.close()


def test_builder_skips_normalized_tag_source(tmp_path: Path) -> None:
    sources_dir = tmp_path / "sources"

    tags_v4_path = sources_dir / "tags_v4.db"
    _create_minimal_tags_v4_db(tags_v4_path)

    csv_root = sources_dir / "TagDB_DataSource_CSV" / "A"
    csv_root.mkdir(parents=True, exist_ok=True)

    # NORMALIZEDっぽい（スペース区切り）tag列しか持たないソース
    normalized_file = csv_root / "normalized_source.csv"
    normalized_file.write_text(
        "tag\nnormalized only tag\nanother normalized tag\n",
        encoding="utf-8",
    )

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
        # スキップされているので TAGS に入らない
        tags = {r[0] for r in conn.execute("SELECT tag FROM TAGS").fetchall()}
        assert "normalized only tag" not in tags
        assert "another normalized tag" not in tags
    finally:
        conn.close()

    skipped_path = report_dir / "skipped_sources.tsv"
    assert skipped_path.exists()
    txt = skipped_path.read_text(encoding="utf-8")
    assert "normalized_source.csv" in txt
    assert "normalized_skipped" in txt


def test_builder_allows_normalized_source_with_override(tmp_path: Path) -> None:
    sources_dir = tmp_path / "sources"

    tags_v4_path = sources_dir / "tags_v4.db"
    _create_minimal_tags_v4_db(tags_v4_path)

    csv_root = sources_dir / "TagDB_DataSource_CSV" / "A"
    csv_root.mkdir(parents=True, exist_ok=True)

    normalized_file = csv_root / "normalized_source.csv"
    normalized_file.write_text(
        "tag\nnormalized only tag\n",
        encoding="utf-8",
    )

    overrides_path = tmp_path / "column_type_overrides.json"
    overrides_path.write_text(
        json.dumps({str(normalized_file): {"tag": "source"}}),
        encoding="utf-8",
    )

    output_db = tmp_path / "out.db"
    report_dir = tmp_path / "reports"

    build_dataset(
        output_path=output_db,
        sources_dir=sources_dir,
        version="test",
        report_dir=report_dir,
        overrides_path=overrides_path,
        overwrite=True,
    )

    conn = sqlite3.connect(output_db)
    try:
        tags = {r[0] for r in conn.execute("SELECT tag FROM TAGS").fetchall()}
        assert "normalized only tag" in tags
    finally:
        conn.close()
