import sqlite3
from genai_tag_db_dataset_builder.builder import (
    _delete_ja_translations_by_value_list,
    _delete_translations_ascii_only_for_languages,
    _delete_translations_missing_required_script,
    _normalize_language_value,
    _split_comma_delimited_translations,
)


def _create_minimal_translations_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE TAG_TRANSLATIONS (
            translation_id INTEGER PRIMARY KEY AUTOINCREMENT,
            tag_id INTEGER NOT NULL,
            language TEXT NOT NULL,
            translation TEXT NOT NULL
        )
        """
    )
    return conn


def _create_translations_db_with_timestamps() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE TAG_TRANSLATIONS (
            translation_id INTEGER PRIMARY KEY AUTOINCREMENT,
            tag_id INTEGER NOT NULL,
            language TEXT NOT NULL,
            translation TEXT NOT NULL,
            created_at TEXT,
            updated_at TEXT
        )
        """
    )
    return conn


def test_delete_ja_translations_by_value_list_deletes_only_ja() -> None:
    conn = _create_minimal_translations_db()
    conn.executemany(
        "INSERT INTO TAG_TRANSLATIONS (tag_id, language, translation) VALUES (?, ?, ?)",
        [
            (1, "ja", "新年快樂"),
            (1, "zh", "新年快樂"),
            (2, "ja", "猫耳"),
            (3, "en", "hello"),
        ],
    )
    conn.commit()

    deleted = _delete_ja_translations_by_value_list(conn, values=["新年快樂", "not_exists"])
    assert deleted == 1
    remaining = conn.execute(
        "SELECT language, translation FROM TAG_TRANSLATIONS ORDER BY translation_id"
    ).fetchall()
    assert ("zh", "新年快樂") in remaining
    assert ("ja", "新年快樂") not in remaining


def test_delete_ja_translations_by_value_list_empty_is_noop() -> None:
    conn = _create_minimal_translations_db()
    deleted = _delete_ja_translations_by_value_list(conn, values=[])
    assert deleted == 0


def test_normalize_language_value_maps_names() -> None:
    assert _normalize_language_value("japanese") == "ja"
    assert _normalize_language_value("zh-Hant") == "zh"


def test_delete_translations_ascii_only_for_languages() -> None:
    conn = _create_minimal_translations_db()
    conn.executemany(
        "INSERT INTO TAG_TRANSLATIONS (tag_id, language, translation) VALUES (?, ?, ?)",
        [
            (1, "ja", "hello"),
            (2, "ja", "猫耳"),
            (3, "zh", "test"),
            (4, "zh", "中文"),
            (5, "en", "hello"),
        ],
    )
    conn.commit()

    deleted = _delete_translations_ascii_only_for_languages(conn, languages={"ja", "zh", "ko"})
    # ja: 'hello' deleted, zh: 'test' deleted, others remain
    assert deleted == 2
    remaining = conn.execute(
        "SELECT language, translation FROM TAG_TRANSLATIONS ORDER BY translation_id"
    ).fetchall()
    assert ("ja", "hello") not in remaining
    assert ("zh", "test") not in remaining
    assert ("ja", "猫耳") in remaining
    assert ("zh", "中文") in remaining
    assert ("en", "hello") in remaining


def test_delete_translations_missing_required_script_ja() -> None:
    conn = _create_minimal_translations_db()
    conn.executemany(
        "INSERT INTO TAG_TRANSLATIONS (tag_id, language, translation) VALUES (?, ?, ?)",
        [
            (1, "ja", "猫耳"),
            (2, "ja", "！！"),
            (3, "ja", "abc"),
            (4, "ja", "Digimon Universe：Appli Monsters"),
            (5, "ja", "ねこみみ"),
        ],
    )
    conn.commit()
    deleted = _delete_translations_missing_required_script(conn, language="ja")
    assert deleted == 3
    remaining = {r[0] for r in conn.execute("SELECT translation FROM TAG_TRANSLATIONS").fetchall()}
    assert remaining == {"猫耳", "ねこみみ"}


def test_split_comma_delimited_translations_removes_empty_entries() -> None:
    conn = _create_translations_db_with_timestamps()
    conn.executemany(
        "INSERT INTO TAG_TRANSLATIONS (tag_id, language, translation, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?)",
        [
            (1, "ja", ",アークナイツ,アークナイツバトルイラコン", "2024-01-01", "2024-01-02"),
            (2, "ja", "魔女", "2024-01-01", "2024-01-02"),
        ],
    )
    conn.commit()

    deleted = _split_comma_delimited_translations(conn)
    assert deleted == 1

    remaining = conn.execute(
        "SELECT tag_id, language, translation FROM TAG_TRANSLATIONS ORDER BY translation_id"
    ).fetchall()
    assert (2, "ja", "魔女") in remaining
    assert (1, "ja", "アークナイツ") in remaining
    assert (1, "ja", "アークナイツバトルイラコン") in remaining
    assert not any(r[2].startswith(",") for r in remaining)


def test_split_comma_delimited_translations_replaces_single_part() -> None:
    conn = _create_translations_db_with_timestamps()
    conn.executemany(
        "INSERT INTO TAG_TRANSLATIONS (tag_id, language, translation, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?)",
        [
            (1, "ja", ",崩壊", "2024-01-01", "2024-01-02"),
            (2, "ja", "つくよみちゃん,", "2024-01-01", "2024-01-02"),
        ],
    )
    conn.commit()

    deleted = _split_comma_delimited_translations(conn)
    assert deleted == 2

    remaining = conn.execute(
        "SELECT tag_id, language, translation FROM TAG_TRANSLATIONS ORDER BY translation_id"
    ).fetchall()
    assert (1, "ja", "崩壊") in remaining
    assert (2, "ja", "つくよみちゃん") in remaining
    assert not any(r[2].startswith(",") or r[2].endswith(",") for r in remaining)
