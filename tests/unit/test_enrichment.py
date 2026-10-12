from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from genai_tag_db_dataset_builder.builder import build_dataset
from genai_tag_db_dataset_builder.core.database import SCHEMA_SQL
from genai_tag_db_dataset_builder.core.enrichment import SCHEMA, apply_enrichment, load_bundle


@pytest.fixture
def conn() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.execute("PRAGMA foreign_keys=ON")
    for statement in SCHEMA_SQL:
        connection.execute(statement)
    connection.execute("INSERT INTO TAGS(tag_id, source_tag, tag) VALUES (1,'midriff','midriff')")
    connection.execute("INSERT INTO TAG_TRANSLATIONS(tag_id,language,translation) VALUES (1,'ja','お腹')")
    connection.commit()
    yield connection
    connection.close()


def make_bundle(tmp_path: Path) -> Path:
    ref = {"source_path": "sheet/elements.tsv", "source_row": 2}
    data = {
        "schema_version": 1,
        "bundle_id": "test",
        "tags": [
            {
                "tag": "midriff",
                "source_tag": "midriff",
                "translations": ["中程度"],
                "source_references": [ref],
            },
            {
                "tag": "dynamic angle",
                "source_tag": "dynamic_angle",
                "translations": ["躍動的な角度", "ダイナミックな角度", "ASCII"],
                "source_references": [ref],
            },
        ],
        "categories": [
            {
                "category_id": "root",
                "namespace": "sheet",
                "parent_id": None,
                "label": "構図",
                "depth": 1,
                "path": ["構図"],
            },
            {
                "category_id": "child",
                "namespace": "sheet",
                "parent_id": "root",
                "label": "角度",
                "depth": 2,
                "path": ["構図", "角度"],
            },
        ],
        "assignments": [
            {"tag": "midriff", "category_id": "root", "source_references": [ref]},
            {"tag": "dynamic angle", "category_id": "child", "source_references": [ref]},
        ],
    }
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def test_additive_import_and_idempotence(conn: sqlite3.Connection, tmp_path: Path) -> None:
    bundle_path = make_bundle(tmp_path)
    result = apply_enrichment(conn, bundle_path, tmp_path / "report")
    assert (
        result.added_tags,
        result.added_translations,
        result.added_categories,
        result.added_memberships,
    ) == (1, 2, 2, 2)
    assert conn.execute("SELECT translation FROM TAG_TRANSLATIONS WHERE tag_id=1").fetchall() == [("お腹",)]
    assert conn.execute("SELECT format_id,type_id,alias FROM TAG_STATUS").fetchall() == [(0, 0, 0)]
    assert conn.execute("SELECT COUNT(*) FROM TAG_CATEGORY_SOURCES").fetchone()[0] == 2
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    repeat = apply_enrichment(conn, bundle_path)
    assert repeat.db_changes == 0
    assert (
        repeat.added_tags
        == repeat.added_translations
        == repeat.added_categories
        == repeat.added_memberships
        == 0
    )
    assert (tmp_path / "report" / "tag_table_test.json").is_file()


def test_translated_alias_target_prevents_fill(conn: sqlite3.Connection, tmp_path: Path) -> None:
    conn.execute("INSERT INTO TAG_FORMATS VALUES (1,'danbooru','')")
    conn.execute("INSERT INTO TAG_TYPE_NAME VALUES (0,'unknown','')")
    conn.execute("INSERT INTO TAG_TYPE_FORMAT_MAPPING VALUES (1,0,0,'')")
    conn.execute("INSERT INTO TAGS(tag_id,source_tag,tag) VALUES (2,'dynamic_angle','dynamic angle')")
    conn.execute(
        "INSERT INTO TAG_STATUS(tag_id,format_id,type_id,alias,preferred_tag_id) VALUES (2,1,0,1,1)"
    )
    result = apply_enrichment(conn, make_bundle(tmp_path))
    assert result.added_translations == 0
    assert conn.execute("SELECT * FROM TAG_TRANSLATIONS WHERE tag_id=2").fetchall() == []
    assert {note["reason"] for note in result.translation_notes} == {"existing_ja", "alias_target_ja"}


@pytest.mark.parametrize(
    "change", ["parent", "depth", "duplicate", "assignment", "source_path", "syntax", "version"]
)
def test_invalid_bundle_does_not_mutate(conn: sqlite3.Connection, tmp_path: Path, change: str) -> None:
    path = make_bundle(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if change == "parent":
        data["categories"][1]["parent_id"] = "missing"
    elif change == "depth":
        data["categories"][1]["depth"] = 3
    elif change == "duplicate":
        data["tags"].append(data["tags"][0])
    elif change == "assignment":
        data["assignments"][0]["tag"] = "missing"
    elif change == "source_path":
        data["tags"][0]["source_references"][0]["source_path"] = "C:/private/file.tsv"
    elif change == "syntax":
        data["tags"][0]["tag"] = "red hair, long hair"
    else:
        data["schema_version"] = 2
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        apply_enrichment(conn, path)
    assert conn.execute("SELECT COUNT(*) FROM TAGS").fetchone()[0] == 1
    assert conn.execute("SELECT name FROM sqlite_master WHERE name='TAG_CATEGORIES'").fetchall() == []


def test_late_conflict_rolls_back_tags_and_translations(conn: sqlite3.Connection, tmp_path: Path) -> None:
    for statement in SCHEMA:
        conn.execute(statement)
    conn.execute("INSERT INTO TAG_CATEGORIES VALUES ('root','other',NULL,'Other',1,'[\"Other\"]')")
    conn.commit()
    with pytest.raises(ValueError, match="conflicts"):
        apply_enrichment(conn, make_bundle(tmp_path))
    assert conn.execute("SELECT COUNT(*) FROM TAGS").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM TAG_TRANSLATIONS").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM TAG_CATEGORY_MEMBERSHIPS").fetchone()[0] == 0


def test_disk_off_journal_is_switched_before_rollback(tmp_path: Path) -> None:
    connection = sqlite3.connect(tmp_path / "off.sqlite")
    try:
        for statement in (*SCHEMA_SQL, *SCHEMA):
            connection.execute(statement)
        connection.execute(
            "INSERT INTO TAG_CATEGORIES VALUES ('root','other',NULL,'Other',1,'[\"Other\"]')"
        )
        connection.commit()
        connection.execute("PRAGMA journal_mode=OFF")
        with pytest.raises(ValueError, match="conflicts"):
            apply_enrichment(connection, make_bundle(tmp_path))
        assert connection.execute("SELECT COUNT(*) FROM TAGS").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM TAG_TRANSLATIONS").fetchone()[0] == 0
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "memory"
    finally:
        connection.close()


def test_off_journal_with_pending_transaction_rejected(tmp_path: Path) -> None:
    connection = sqlite3.connect(tmp_path / "pending.sqlite")
    try:
        for statement in SCHEMA_SQL:
            connection.execute(statement)
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("INSERT INTO TAGS(source_tag,tag) VALUES ('old','old')")
        with pytest.raises(ValueError, match="rollback journal"):
            apply_enrichment(connection, make_bundle(tmp_path))
        assert connection.execute("SELECT tag FROM TAGS").fetchall() == [("old",)]
    finally:
        connection.close()


def test_checked_in_bundle_validates() -> None:
    path = Path(__file__).resolve().parents[2] / "data" / "tag_table_bundle.json"
    bundle, digest = load_bundle(path)
    assert len(bundle.tags) > 3000
    assert len(bundle.categories) > 400
    assert len(digest) == 64


@pytest.mark.parametrize("tag", ["tag:+1", r"\(tag:-1\)", "tag:-.5", "tag\tname", "tag\x7fname"])
def test_direct_bundle_rejects_weights_and_controls(tmp_path: Path, tag: str) -> None:
    path = make_bundle(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["tags"][0]["tag"] = tag
    data["assignments"][0]["tag"] = tag
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical single tag"):
        load_bundle(path)


def test_builder_rejects_same_base_and_output(tmp_path: Path) -> None:
    path = tmp_path / "base.sqlite"
    path.write_bytes(b"must remain intact")
    with pytest.raises(ValueError, match="different paths"):
        build_dataset(path, tmp_path, "test", base_db_path=path, overwrite=True)
    assert path.read_bytes() == b"must remain intact"


def test_builder_applies_bundle_to_base_copy(tmp_path: Path) -> None:
    base = tmp_path / "base.sqlite"
    connection = sqlite3.connect(base)
    for statement in SCHEMA_SQL:
        connection.execute(statement)
    connection.commit()
    connection.close()
    output = tmp_path / "output.sqlite"
    build_dataset(
        output,
        tmp_path,
        "test",
        base_db_path=base,
        tag_table_bundles=[make_bundle(tmp_path)],
        enable_site_tags=False,
    )
    with sqlite3.connect(output) as check:
        assert check.execute("SELECT COUNT(*) FROM TAG_CATEGORY_MEMBERSHIPS").fetchone()[0] == 2
        assert check.execute("PRAGMA foreign_key_check").fetchall() == []
    check.close()


def test_repeated_build_keeps_split_translations(tmp_path: Path) -> None:
    bundle_path = make_bundle(tmp_path)
    data = json.loads(bundle_path.read_text(encoding="utf-8"))
    data["tags"][1]["translations"] = ["黒、白"]
    bundle_path.write_text(json.dumps(data), encoding="utf-8")
    first, second = tmp_path / "first.sqlite", tmp_path / "second.sqlite"
    build_dataset(first, tmp_path, "one", tag_table_bundles=[bundle_path], enable_site_tags=False)
    build_dataset(
        second, tmp_path, "two", base_db_path=first, tag_table_bundles=[bundle_path], enable_site_tags=False
    )
    translations = []
    for path in (first, second):
        connection = sqlite3.connect(path)
        try:
            translations.append(
                connection.execute(
                    "SELECT tag_id,language,translation FROM TAG_TRANSLATIONS ORDER BY tag_id,language,translation"
                ).fetchall()
            )
        finally:
            connection.close()
    assert translations[0] == translations[1]
    assert {row[2] for row in translations[0] if row[0] == 1} == {"黒", "白"}
