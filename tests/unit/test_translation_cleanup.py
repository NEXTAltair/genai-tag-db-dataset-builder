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
            (2, "ja", "\uff01\uff01"),
            (3, "ja", "abc"),
            (4, "ja", "Digimon Universe\uff1aAppli Monsters"),
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


def test_reclassify_chinese_ja_translations_as_zh_moves_and_dedupes() -> None:
    """language='ja' の中国語 (簡体字) を zh へ再分類。zh 重複は削除 (LoRAIro #1213)。"""
    from genai_tag_db_dataset_builder.builder import _reclassify_chinese_ja_translations_as_zh

    conn = _create_minimal_translations_db()
    conn.executemany(
        "INSERT INTO TAG_TRANSLATIONS (tag_id, language, translation) VALUES (?, ?, ?)",
        [
            (1, "ja", "发饰"),  # 简体 '发' → zh へ move
            (2, "ja", "电话"),  # 简体 '电' → zh へ move
            (3, "ja", "猫耳"),  # 日本語 (簡体字専用文字なし) → ja のまま
            (4, "ja", "独角兽"),  # 简体 '兽' → zh へ move (import 元の language 誤り)
            (5, "zh", "发饰"),  # 既存 zh
            (5, "ja", "发饰"),  # tag_id=5 は zh 重複あり → 削除
        ],
    )
    conn.commit()

    moved = _reclassify_chinese_ja_translations_as_zh(conn)
    assert moved == 4  # 3 move + 1 delete

    remaining = conn.execute(
        "SELECT tag_id, language, translation FROM TAG_TRANSLATIONS ORDER BY tag_id, language"
    ).fetchall()
    # 日本語はそのまま
    assert (3, "ja", "猫耳") in remaining
    # 中国語は zh へ
    assert (1, "zh", "发饰") in remaining
    assert (2, "zh", "电话") in remaining
    assert (4, "zh", "独角兽") in remaining
    # tag_id=5 の ja 重複は削除され、zh 1 件のみ
    tag5 = [r for r in remaining if r[0] == 5]
    assert tag5 == [(5, "zh", "发饰")]
    # ja に簡体字専用文字を含む行は残っていない
    assert not any(r[1] == "ja" and r[2] in ("发饰", "电话", "独角兽") for r in remaining)


def test_delete_underscore_alias_translations_removes_only_underscore_en() -> None:
    """先頭 '_' の en 翻訳 (エイリアス表記) のみ削除する (LoRAIro #1213)。"""
    from genai_tag_db_dataset_builder.builder import _delete_underscore_alias_translations

    conn = _create_minimal_translations_db()
    conn.executemany(
        "INSERT INTO TAG_TRANSLATIONS (tag_id, language, translation) VALUES (?, ?, ?)",
        [
            (1, "en", "___sparkles"),  # 削除対象
            (2, "en", "__1girl"),  # 削除対象
            (3, "en", "sparkles"),  # 通常訳 → 残す
            (4, "ja", "_test"),  # ja の '_' は対象外 → 残す
            (5, "en", "a_b"),  # 中間 '_' は対象外 → 残す
        ],
    )
    conn.commit()

    deleted = _delete_underscore_alias_translations(conn)
    assert deleted == 2

    remaining = conn.execute(
        "SELECT language, translation FROM TAG_TRANSLATIONS ORDER BY translation_id"
    ).fetchall()
    assert ("en", "___sparkles") not in remaining
    assert ("en", "__1girl") not in remaining
    assert ("en", "sparkles") in remaining
    assert ("ja", "_test") in remaining
    assert ("en", "a_b") in remaining


def test_reclassify_chinese_ja_empty_is_noop() -> None:
    from genai_tag_db_dataset_builder.builder import _reclassify_chinese_ja_translations_as_zh

    conn = _create_minimal_translations_db()
    assert _reclassify_chinese_ja_translations_as_zh(conn) == 0
