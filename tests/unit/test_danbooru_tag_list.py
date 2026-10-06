from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from datasets import Dataset

from genai_tag_db_dataset_builder.adapters.hf_translation_adapter import DanbooruTagListAdapter
from genai_tag_db_dataset_builder.builder import build_dataset


def _save_tag_list(tmp_path: Path, name: str, rows: list[dict]) -> str:
    ds = Dataset.from_dict({k: [r[k] for r in rows] for k in rows[0]})
    path = tmp_path / name
    ds.save_to_disk(path.as_posix())
    return path.as_posix()


def _build(tmp_path: Path, tag_list: str, name: str, *, base_db: Path | None = None) -> Path:
    (tmp_path / "sources").mkdir(exist_ok=True)
    out = tmp_path / name
    build_dataset(
        output_path=out,
        sources_dir=tmp_path / "sources",
        version="test",
        report_dir=tmp_path / f"report_{name}",
        overwrite=True,
        base_db_path=base_db,
        hf_zh_translation_datasets=[tag_list],
        hf_danbooru_tag_list_datasets=[tag_list],
    )
    return out


def test_adapter_reads_tag_category_count_and_drops_invalid_rows(tmp_path: Path) -> None:
    path = _save_tag_list(
        tmp_path,
        "ds",
        [
            {"tag": "1girl", "category": 0, "count": 10, "zh": "一个女孩"},
            {"tag": "bad_category", "category": 9, "count": 1, "zh": "x"},
            {"tag": "", "category": 0, "count": 1, "zh": "x"},
        ],
    )
    df = DanbooruTagListAdapter(path).read()
    assert df.to_dicts() == [{"source_tag": "1girl", "category": 0, "count": 10}]


def test_build_creates_missing_tags_with_type_count_and_translation(tmp_path: Path) -> None:
    path = _save_tag_list(
        tmp_path,
        "ds",
        [
            {"tag": "new_character_(series)", "category": 4, "count": 1234, "zh": "新角色"},
            {"tag": "rating:general", "category": 5, "count": 99, "zh": "全年龄"},
        ],
    )
    out = _build(tmp_path, path, "out.db")

    conn = sqlite3.connect(out)
    try:
        row = conn.execute(
            "SELECT t.tag_id, s.type_id, s.alias, s.preferred_tag_id, u.count "
            "FROM TAGS t JOIN TAG_STATUS s ON s.tag_id = t.tag_id AND s.format_id = 1 "
            "JOIN TAG_USAGE_COUNTS u ON u.tag_id = t.tag_id AND u.format_id = 1 "
            "WHERE t.tag = ?",
            ("new character \\(series\\)",),
        ).fetchone()
        assert row is not None
        tag_id, type_id, alias, preferred, count = row
        assert (type_id, alias, preferred, count) == (4, 0, tag_id, 1234)
        zh = conn.execute(
            "SELECT translation FROM TAG_TRANSLATIONS WHERE tag_id = ? AND language = 'zh'", (tag_id,)
        ).fetchall()
        assert [r[0] for r in zh] == ["新角色"]
        assert conn.execute(
            "SELECT type_id FROM TAG_STATUS WHERE tag_id = (SELECT tag_id FROM TAGS WHERE tag = 'rating:general') AND format_id = 1"
        ).fetchone() == (5,)
    finally:
        conn.close()


def test_existing_danbooru_status_and_alias_are_left_untouched(tmp_path: Path) -> None:
    first = _save_tag_list(
        tmp_path,
        "first",
        [
            {"tag": "existing", "category": 0, "count": 500, "zh": "既存"},
            {"tag": "alias_tag", "category": 0, "count": 5, "zh": "别名"},
        ],
    )
    base = _build(tmp_path, first, "base.db")
    conn = sqlite3.connect(base)
    try:
        ids = dict(conn.execute("SELECT tag, tag_id FROM TAGS").fetchall())
        # alias_tag を existing の別名にする
        conn.execute("DELETE FROM TAG_USAGE_COUNTS WHERE tag_id = ?", (ids["alias tag"],))
        conn.execute(
            "UPDATE TAG_STATUS SET alias = 1, preferred_tag_id = ? WHERE tag_id = ? AND format_id = 1",
            (ids["existing"], ids["alias tag"]),
        )
        conn.commit()
    finally:
        conn.close()

    # 2 回目: 種別・件数が違う同じタグ + 新規タグ
    second = _save_tag_list(
        tmp_path,
        "second",
        [
            {"tag": "existing", "category": 4, "count": 9999, "zh": "既存"},
            {"tag": "alias_tag", "category": 4, "count": 9999, "zh": "别名"},
            {"tag": "brand_new", "category": 3, "count": 7, "zh": "全新"},
        ],
    )
    out = _build(tmp_path, second, "out.db", base_db=base)

    conn = sqlite3.connect(out)
    try:
        ids = dict(conn.execute("SELECT tag, tag_id FROM TAGS").fetchall())
        status = {
            tag: conn.execute(
                "SELECT type_id, alias, preferred_tag_id FROM TAG_STATUS WHERE tag_id = ? AND format_id = 1",
                (ids[tag],),
            ).fetchone()
            for tag in ("existing", "alias tag", "brand new")
        }
        assert status["existing"] == (0, 0, ids["existing"])  # 種別は上書きしない
        assert status["alias tag"] == (0, 1, ids["existing"])  # 別名関係は維持
        assert status["brand new"] == (3, 0, ids["brand new"])
        assert conn.execute(
            "SELECT count FROM TAG_USAGE_COUNTS WHERE tag_id = ? AND format_id = 1", (ids["existing"],)
        ).fetchone() == (500,)  # 既存の件数は上書きしない
        assert conn.execute(
            "SELECT count(*) FROM TAG_USAGE_COUNTS WHERE tag_id = ? AND format_id = 1", (ids["alias tag"],)
        ).fetchone() == (0,)
    finally:
        conn.close()


@pytest.mark.parametrize("flag", [False, True])
def test_tag_list_is_ignored_unless_requested(tmp_path: Path, flag: bool) -> None:
    path = _save_tag_list(
        tmp_path, "ds", [{"tag": "only_translation", "category": 0, "count": 1, "zh": "仅翻译"}]
    )
    (tmp_path / "sources").mkdir(exist_ok=True)
    out = tmp_path / "out.db"
    build_dataset(
        output_path=out,
        sources_dir=tmp_path / "sources",
        version="test",
        report_dir=tmp_path / "report",
        overwrite=True,
        hf_zh_translation_datasets=[path],
        hf_danbooru_tag_list_datasets=[path] if flag else None,
    )
    conn = sqlite3.connect(out)
    try:
        n = conn.execute("SELECT count(*) FROM TAGS").fetchone()[0]
    finally:
        conn.close()
    # 要求しなければ翻訳だけ (タグが無いので何も入らない)、要求すればタグが作られる
    assert n == (1 if flag else 0)
