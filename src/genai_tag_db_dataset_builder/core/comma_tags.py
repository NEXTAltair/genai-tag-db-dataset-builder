"""カンマを含むタグを Danbooru 形式の alias へ対応付ける.

カンマはプロンプトの区切り文字なので、カンマ付きタグをそのまま挿入すると 2 つのタグに割れる。
pixiv などの表記は残したまま、Danbooru で通じるタグへの alias を追加する。
"""

from __future__ import annotations

import csv
import re
import sqlite3
from pathlib import Path

import yaml

_DANBOORU_FORMAT_ID = 1
_GENERAL_TYPE_ID = 0
_MIN_TARGET_USAGE = 10
_INNER_COMMA = re.compile(r"[a-z0-9]{2,},\s*[a-z0-9]{2,}")


def load_comma_tag_overrides(path: Path | str | None) -> dict[str, str]:
    """``overrides/comma_tags.yml`` を読み込む（無ければ空）."""
    if path is None or not Path(path).exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return dict((data or {}).get("comma_tags") or {})


def _without_commas(tag: str) -> str:
    return " ".join(tag.replace(",", " ").split())


def _rule_target(tag: str, usage_of_target: int) -> bool:
    """カンマを消した同名タグへの対応を機械で採用できるなら True."""
    if tag.startswith(","):
        return False
    if tag.rstrip(", ") == _without_commas(tag):
        return True
    return (
        usage_of_target >= _MIN_TARGET_USAGE
        and _INNER_COMMA.search(tag) is not None
        and len(_without_commas(tag).split()) >= 3
    )


def add_comma_tag_aliases(conn: sqlite3.Connection, overrides: dict[str, str]) -> list[dict[str, str]]:
    """カンマ付きタグを Danbooru 形式の alias として登録する.

    Returns:
        登録した対応の一覧（tag, target, reason）。
    """
    tag_ids = dict(conn.execute("SELECT tag, tag_id FROM TAGS"))
    status = {
        int(tag_id): (bool(alias), int(preferred_tag_id), int(type_id))
        for tag_id, alias, preferred_tag_id, type_id in conn.execute(
            "SELECT tag_id, alias, preferred_tag_id, type_id FROM TAG_STATUS WHERE format_id = ?",
            (_DANBOORU_FORMAT_ID,),
        )
    }
    usage = dict(
        conn.execute(
            "SELECT tag_id, count FROM TAG_USAGE_COUNTS WHERE format_id = ?", (_DANBOORU_FORMAT_ID,)
        )
    )

    added: list[dict[str, str]] = []
    for tag, tag_id in [(tag, tag_id) for tag, tag_id in tag_ids.items() if "," in tag]:
        reason = "override"
        target_name = overrides.get(tag)
        if target_name is None:
            reason = "rule"
            target_name = _without_commas(tag)
        target_id = tag_ids.get(target_name)

        if target_id is None:
            if reason == "rule":
                continue
            target_id = conn.execute(
                "INSERT INTO TAGS (source_tag, tag) VALUES (?, ?) RETURNING tag_id",
                (target_name.replace(" ", "_"), target_name),
            ).fetchone()[0]
            tag_ids[target_name] = target_id
        if target_id not in status:
            if reason == "rule":
                continue
            conn.execute(
                "INSERT INTO TAG_STATUS (tag_id, format_id, type_id, alias, preferred_tag_id) VALUES (?, ?, ?, 0, ?)",
                (target_id, _DANBOORU_FORMAT_ID, _GENERAL_TYPE_ID, target_id),
            )
            status[target_id] = (False, target_id, _GENERAL_TYPE_ID)

        is_alias, preferred_tag_id, type_id = status[target_id]
        if is_alias:
            target_id = preferred_tag_id
            if target_id not in status:
                continue
            type_id = status[target_id][2]
        if target_id == tag_id:
            continue
        if reason == "rule" and not _rule_target(tag, int(usage.get(target_id, 0))):
            continue

        conn.execute(
            "INSERT INTO TAG_STATUS (tag_id, format_id, type_id, alias, preferred_tag_id) VALUES (?, ?, ?, 1, ?) "
            "ON CONFLICT(tag_id, format_id) DO UPDATE SET "
            "type_id = excluded.type_id, alias = 1, preferred_tag_id = excluded.preferred_tag_id",
            (tag_id, _DANBOORU_FORMAT_ID, type_id, target_id),
        )
        status[tag_id] = (True, target_id, type_id)
        added.append({"tag": tag, "target": target_name, "reason": reason})
    conn.commit()
    return added


def write_comma_tag_aliases_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["tag", "target", "reason"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
