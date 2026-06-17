"""非e621 カテゴリ prefix 逆転是正 + orphan prefix タグ改名のテスト。

`_repair_non_e621_prefix_preferred` / `_rename_orphan_prefixed_tags` が
copyright:/character:/species: 等 全カテゴリ prefix を対象に、preferred 逆転を
prefix 無しの canonical へ寄せ、clean 等価が無い prefix タグを改名することを検証する。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from genai_tag_db_dataset_builder.builder import (
    _rename_orphan_prefixed_tags,
    _repair_non_e621_prefix_preferred,
)
from genai_tag_db_dataset_builder.core.database import create_database

pytestmark = pytest.mark.unit


def _make_db(tmp_path: Path) -> sqlite3.Connection:
    db_path = tmp_path / "unified.sqlite"
    create_database(db_path)
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        INSERT INTO TAG_FORMATS (format_id, format_name) VALUES (1, 'danbooru'), (2, 'e621');
        INSERT INTO TAGS (tag_id, source_tag, tag) VALUES
            (6360, 'vtuber', 'vtuber'),
            (6363, 'copyright:vtuber', 'copyright:vtuber'),
            (7000, 'copyright:onlyprefixed', 'copyright:onlyprefixed');
        -- danbooru(1): vtuber が copyright:vtuber を preferred に持つ逆転 (alias=1)
        INSERT INTO TAG_STATUS (tag_id, format_id, type_id, alias, preferred_tag_id) VALUES
            (6360, 1, 0, 1, 6363),
            (6363, 1, 0, 0, 6363),
            (7000, 1, 0, 0, 7000);
        -- e621(2): 触らない対象 (format_id=2 は除外)
        INSERT INTO TAG_STATUS (tag_id, format_id, type_id, alias, preferred_tag_id) VALUES
            (6360, 2, 3, 0, 6360);
        """
    )
    conn.commit()
    return conn


def _status(conn: sqlite3.Connection, tag_id: int, format_id: int) -> tuple[int, int]:
    row = conn.execute(
        "SELECT alias, preferred_tag_id FROM TAG_STATUS WHERE tag_id = ? AND format_id = ?",
        (tag_id, format_id),
    ).fetchone()
    return int(row[0]), int(row[1])


def test_repair_fixes_copyright_prefix_inversion(tmp_path: Path) -> None:
    """copyright: 逆転が canonical (prefix 無し) へ是正される (meta:/artist: 以外も対象)。"""
    conn = _make_db(tmp_path)
    try:
        repairs = _repair_non_e621_prefix_preferred(conn)

        # vtuber(danbooru) は canonical 化 (alias=0, preferred=self)
        assert _status(conn, 6360, 1) == (0, 6360)
        # copyright:vtuber(danbooru) は vtuber の alias 化
        assert _status(conn, 6363, 1) == (1, 6360)
        # e621(format_id=2) は不変
        assert _status(conn, 6360, 2) == (0, 6360)
        reasons = {r["reason"] for r in repairs}
        assert "prefix_to_base" in reasons
    finally:
        conn.close()


def test_orphan_prefixed_tag_renamed_when_no_base(tmp_path: Path) -> None:
    """clean 等価が無い prefix タグは prefix 除去で改名される。"""
    conn = _make_db(tmp_path)
    try:
        _repair_non_e621_prefix_preferred(conn)
        renames = _rename_orphan_prefixed_tags(conn)

        # clean 等価ありの copyright:vtuber は改名しない (alias 解決に委ねる)
        assert conn.execute("SELECT tag FROM TAGS WHERE tag_id = 6363").fetchone()[0] == "copyright:vtuber"
        # clean 等価なしの copyright:onlyprefixed は改名される
        assert conn.execute("SELECT tag FROM TAGS WHERE tag_id = 7000").fetchone()[0] == "onlyprefixed"
        assert any(r["new_tag"] == "onlyprefixed" for r in renames)
    finally:
        conn.close()
