"""カンマ付きタグの Danbooru alias のテスト."""

import sqlite3
from pathlib import Path

from genai_tag_db_dataset_builder.builder import _flatten_alias_chains
from genai_tag_db_dataset_builder.core.comma_tags import add_comma_tag_aliases
from genai_tag_db_dataset_builder.core.database import create_schema
from genai_tag_db_dataset_builder.core.master_data import initialize_master_data


def _db(path: Path, tags: list[str], danbooru: dict[str, tuple[int, int]]) -> sqlite3.Connection:
    sqlite3.connect(path).close()
    create_schema(path)
    initialize_master_data(path)
    conn = sqlite3.connect(path)
    for tag in tags:
        conn.execute("INSERT INTO TAGS (source_tag, tag) VALUES (?, ?)", (tag, tag))
    ids = dict(conn.execute("SELECT tag, tag_id FROM TAGS"))
    for tag, (type_id, count) in danbooru.items():
        conn.execute(
            "INSERT INTO TAG_STATUS (tag_id, format_id, type_id, alias, preferred_tag_id) VALUES (?, 1, ?, 0, ?)",
            (ids[tag], type_id, ids[tag]),
        )
        conn.execute(
            "INSERT INTO TAG_USAGE_COUNTS (tag_id, format_id, count) VALUES (?, 1, ?)", (ids[tag], count)
        )
    conn.commit()
    return conn


def _preferred(conn: sqlite3.Connection, tag: str) -> str | None:
    row = conn.execute(
        "SELECT p.tag FROM TAG_STATUS s JOIN TAGS t ON t.tag_id = s.tag_id JOIN TAGS p ON p.tag_id = s.preferred_tag_id "
        "WHERE t.tag = ? AND s.format_id = 1 AND s.alias = 1",
        (tag,),
    ).fetchone()
    return row[0] if row else None


def test_rules_map_trailing_and_inner_commas(tmp_path: Path) -> None:
    conn = _db(
        tmp_path / "db.sqlite",
        [
            "nude,",
            "nude",
            "ass",
            ",ass",
            "a long title, part two",
            "a long title part two",
            "ab, cd",
            "ab cd",
        ],
        {"nude": (0, 100), "ass": (0, 100), "a long title part two": (3, 20), "ab cd": (0, 100)},
    )

    added = add_comma_tag_aliases(conn, {})

    assert {row["tag"] for row in added} == {"nude,", "a long title, part two"}
    assert _preferred(conn, "nude,") == "nude"
    assert _preferred(conn, "a long title, part two") == "a long title part two"
    assert _preferred(conn, ",ass") is None
    assert _preferred(conn, "ab, cd") is None


def test_rule_skips_low_usage_inner_comma(tmp_path: Path) -> None:
    conn = _db(
        tmp_path / "db.sqlite", ["a long title, here", "a long title here"], {"a long title here": (0, 9)}
    )

    assert add_comma_tag_aliases(conn, {}) == []


def test_trailing_comma_ignores_usage(tmp_path: Path) -> None:
    conn = _db(tmp_path / "db.sqlite", ["youtuber,", "youtuber"], {"youtuber": (0, 0)})

    add_comma_tag_aliases(conn, {})

    assert _preferred(conn, "youtuber,") == "youtuber"


def test_override_resolves_alias_target_and_keeps_type(tmp_path: Path) -> None:
    conn = _db(
        tmp_path / "db.sqlite",
        ["title, en", "romaji", "old name"],
        {"romaji": (3, 100)},
    )
    ids = dict(conn.execute("SELECT tag, tag_id FROM TAGS"))
    conn.execute(
        "INSERT INTO TAG_STATUS (tag_id, format_id, type_id, alias, preferred_tag_id) VALUES (?, 1, 3, 1, ?)",
        (ids["old name"], ids["romaji"]),
    )

    add_comma_tag_aliases(conn, {"title, en": "old name"})

    assert _preferred(conn, "title, en") == "romaji"
    assert conn.execute(
        "SELECT type_id FROM TAG_STATUS WHERE tag_id = ? AND format_id = 1", (ids["title, en"],)
    ).fetchone() == (3,)


def test_incoming_aliases_follow_converted_comma_tag(tmp_path: Path) -> None:
    conn = _db(
        tmp_path / "db.sqlite",
        ["old title", "a long title, here", "a long title here"],
        {"a long title, here": (3, 0), "a long title here": (3, 20)},
    )
    ids = dict(conn.execute("SELECT tag, tag_id FROM TAGS"))
    conn.execute(
        "INSERT INTO TAG_STATUS (tag_id, format_id, type_id, alias, preferred_tag_id) VALUES (?, 1, 3, 1, ?)",
        (ids["old title"], ids["a long title, here"]),
    )

    add_comma_tag_aliases(conn, {})
    _flatten_alias_chains(conn)

    assert _preferred(conn, "old title") == "a long title here"
    assert _preferred(conn, "a long title, here") == "a long title here"


def test_override_can_target_multi_hop_alias(tmp_path: Path) -> None:
    conn = _db(tmp_path / "db.sqlite", ["title, en", "first", "second", "romaji"], {"romaji": (3, 100)})
    ids = dict(conn.execute("SELECT tag, tag_id FROM TAGS"))
    conn.executemany(
        "INSERT INTO TAG_STATUS (tag_id, format_id, type_id, alias, preferred_tag_id) VALUES (?, 1, 3, 1, ?)",
        [(ids["first"], ids["second"]), (ids["second"], ids["romaji"])],
    )

    add_comma_tag_aliases(conn, {"title, en": "first"})
    _flatten_alias_chains(conn)

    assert _preferred(conn, "title, en") == "romaji"


def test_rule_checks_usage_of_final_canonical_tag(tmp_path: Path) -> None:
    conn = _db(
        tmp_path / "db.sqlite",
        ["a long title, here", "a long title here", "intermediate", "canonical"],
        {"canonical": (3, 100)},
    )
    ids = dict(conn.execute("SELECT tag, tag_id FROM TAGS"))
    conn.executemany(
        "INSERT INTO TAG_STATUS (tag_id, format_id, type_id, alias, preferred_tag_id) VALUES (?, 1, 3, 1, ?)",
        [(ids["a long title here"], ids["intermediate"]), (ids["intermediate"], ids["canonical"])],
    )

    add_comma_tag_aliases(conn, {})

    assert _preferred(conn, "a long title, here") == "canonical"


def test_target_chain_leading_back_to_source_is_skipped(tmp_path: Path) -> None:
    conn = _db(
        tmp_path / "db.sqlite",
        ["title,", "title", "other title"],
        {"title,": (0, 5)},
    )
    ids = dict(conn.execute("SELECT tag, tag_id FROM TAGS"))
    conn.executemany(
        "INSERT INTO TAG_STATUS (tag_id, format_id, type_id, alias, preferred_tag_id) VALUES (?, 1, 0, 1, ?)",
        [(ids["title"], ids["other title"]), (ids["other title"], ids["title,"])],
    )

    assert add_comma_tag_aliases(conn, {}) == []
    assert conn.execute(
        "SELECT alias, preferred_tag_id FROM TAG_STATUS WHERE tag_id = ?", (ids["title,"],)
    ).fetchone() == (0, ids["title,"])


def test_target_chain_ending_in_cycle_is_skipped(tmp_path: Path) -> None:
    conn = _db(tmp_path / "db.sqlite", ["title,", "title", "other title"], {})
    ids = dict(conn.execute("SELECT tag, tag_id FROM TAGS"))
    conn.executemany(
        "INSERT INTO TAG_STATUS (tag_id, format_id, type_id, alias, preferred_tag_id) VALUES (?, 1, 0, 1, ?)",
        [(ids["title"], ids["other title"]), (ids["other title"], ids["title"])],
    )

    assert add_comma_tag_aliases(conn, {}) == []


def test_override_creates_missing_target(tmp_path: Path) -> None:
    conn = _db(tmp_path / "db.sqlite", ["paws, inc."], {})

    add_comma_tag_aliases(conn, {"paws, inc.": "paws inc."})

    assert _preferred(conn, "paws, inc.") == "paws inc."
    assert conn.execute(
        "SELECT t.source_tag, s.alias, c.count FROM TAGS t JOIN TAG_STATUS s ON s.tag_id = t.tag_id "
        "LEFT JOIN TAG_USAGE_COUNTS c ON c.tag_id = t.tag_id WHERE t.tag = 'paws inc.'"
    ).fetchone() == ("paws_inc.", 0, None)


def test_existing_canonical_comma_tag_becomes_alias(tmp_path: Path) -> None:
    conn = _db(
        tmp_path / "db.sqlite",
        ["yondemasu yo, azazel-san.", "yondemasu yo azazel-san."],
        {"yondemasu yo, azazel-san.": (3, 0), "yondemasu yo azazel-san.": (3, 215)},
    )

    add_comma_tag_aliases(conn, {})

    assert _preferred(conn, "yondemasu yo, azazel-san.") == "yondemasu yo azazel-san."
