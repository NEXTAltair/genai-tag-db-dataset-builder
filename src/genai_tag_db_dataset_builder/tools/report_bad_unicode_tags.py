"""SQLite内の文字化けタグ（U+FFFD 等）を抽出し、site_tags の元データ起因かを調査してTSV出力する。"""

from __future__ import annotations

import argparse
import csv
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BadTagRow:
    format_id: int
    format_name: str
    tag_id: int
    tag: str
    source_tag: str
    alias: int
    preferred_tag_id: int
    deprecated: int
    usage_count: int | None


def _write_tsv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(header)
        for r in rows:
            w.writerow(["" if v is None else v for v in r])


def _to_unicode_escape(s: str) -> str:
    return s.encode("unicode_escape").decode("ascii")


def _fetch_bad_tags(db_path: Path) -> list[BadTagRow]:
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            """
            SELECT
                ts.format_id,
                f.format_name,
                t.tag_id,
                t.tag,
                t.source_tag,
                ts.alias,
                ts.preferred_tag_id,
                ts.deprecated,
                uc.count AS usage_count
            FROM TAG_STATUS ts
            JOIN TAG_FORMATS f ON f.format_id = ts.format_id
            JOIN TAGS t ON t.tag_id = ts.tag_id
            LEFT JOIN TAG_USAGE_COUNTS uc
                ON uc.tag_id = ts.tag_id
                AND uc.format_id = ts.format_id
            WHERE instr(t.tag, char(65533)) > 0
               OR instr(t.source_tag, char(65533)) > 0
            ORDER BY ts.format_id, t.tag_id
            """
        ).fetchall()
        return [
            BadTagRow(
                format_id=int(r["format_id"]),
                format_name=str(r["format_name"]),
                tag_id=int(r["tag_id"]),
                tag=str(r["tag"]),
                source_tag=str(r["source_tag"]),
                alias=int(r["alias"]),
                preferred_tag_id=int(r["preferred_tag_id"]),
                deprecated=int(r["deprecated"]),
                usage_count=None if r["usage_count"] is None else int(r["usage_count"]),
            )
            for r in rows
        ]
    finally:
        con.close()


def _lookup_tag_id_by_exact_tag(db_path: Path, tag: str) -> int | None:
    con = sqlite3.connect(str(db_path))
    try:
        row = con.execute("SELECT tag_id FROM TAGS WHERE tag = ?;", (tag,)).fetchone()
        return None if row is None else int(row[0])
    finally:
        con.close()


def _lookup_tag_by_id(db_path: Path, tag_id: int) -> str | None:
    con = sqlite3.connect(str(db_path))
    try:
        row = con.execute("SELECT tag FROM TAGS WHERE tag_id = ?;", (tag_id,)).fetchone()
        return None if row is None else str(row[0])
    finally:
        con.close()


def _has_tag_status(db_path: Path, tag_id: int, format_id: int) -> bool:
    con = sqlite3.connect(str(db_path))
    try:
        row = con.execute(
            "SELECT 1 FROM TAG_STATUS WHERE tag_id = ? AND format_id = ? LIMIT 1;",
            (tag_id, format_id),
        ).fetchone()
        return row is not None
    finally:
        con.close()


def _format_to_site_dir(format_name: str) -> str | None:
    # deepghs/site_tags のディレクトリ名（ドメイン） -> TAG_FORMATS.format_name のマッピング。
    # ここでは “文字化け調査で問題になった” 最小セットだけを定義する。
    return {
        "sankaku": "chan.sankakucomplex.com",
        "rule34": "rule34.xxx",
        "zerochan": "zerochan.net",
    }.get(format_name)


def _site_tags_schema(sqlite_path: Path) -> tuple[str | None, str | None]:
    """(tags列名, tag_aliases列名) を返す（存在しない場合はNone）。"""
    con = sqlite3.connect(str(sqlite_path))
    try:
        tags_col = None
        try:
            cols = [r[1] for r in con.execute("PRAGMA table_info(tags);").fetchall()]
            if "name" in cols:
                tags_col = "name"
            elif "tag" in cols:
                tags_col = "tag"
        except sqlite3.OperationalError:
            tags_col = None

        alias_col = None
        try:
            cols = [r[1] for r in con.execute("PRAGMA table_info(tag_aliases);").fetchall()]
            if "alias" in cols and "tag" in cols:
                alias_col = "alias"
        except sqlite3.OperationalError:
            alias_col = None

        return tags_col, alias_col
    finally:
        con.close()


def _site_tags_find_origin(site_sqlite: Path, tag_value: str) -> dict[str, Any] | None:
    """site_tags 側で tag_value がどこに存在するかを探し、最小の情報を返す。"""
    tags_col, alias_col = _site_tags_schema(site_sqlite)
    con = sqlite3.connect(str(site_sqlite))
    con.row_factory = sqlite3.Row
    try:
        if tags_col is not None:
            row = con.execute(
                f"SELECT *, hex({tags_col}) AS _hex FROM tags WHERE {tags_col} = ? LIMIT 1;", (tag_value,)
            ).fetchone()
            if row is not None:
                d = dict(row)
                d["_origin"] = f"tags.{tags_col}"
                return d

        if alias_col is not None:
            row = con.execute(
                "SELECT *, hex(alias) AS _hex_alias, hex(tag) AS _hex_tag FROM tag_aliases WHERE alias = ? LIMIT 1;",
                (tag_value,),
            ).fetchone()
            if row is not None:
                d = dict(row)
                d["_origin"] = "tag_aliases.alias"
                return d

        return None
    finally:
        con.close()


def _site_tags_find_alias_by_target_tag(site_sqlite: Path, target_tag_value: str) -> dict[str, Any] | None:
    """tag_aliases.tag 側から逆引きして、U+FFFD を含む alias を探す（zerochan 等の補助用）。"""
    _, alias_col = _site_tags_schema(site_sqlite)
    if alias_col is None:
        return None

    con = sqlite3.connect(str(site_sqlite))
    con.row_factory = sqlite3.Row
    try:
        row = con.execute(
            """
            SELECT *, hex(alias) AS _hex_alias, hex(tag) AS _hex_tag
            FROM tag_aliases
            WHERE lower(tag) = lower(?)
              AND instr(alias, char(65533)) > 0
            LIMIT 1
            """,
            (target_tag_value,),
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["_origin"] = "tag_aliases.tag (fffd alias)"
        return d
    finally:
        con.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Report tags containing U+FFFD and their origins (site_tags).")
    p.add_argument(
        "--db", type=Path, required=True, help="Path to built SQLite DB (e.g. out_db_cc4/*.sqlite)"
    )
    p.add_argument(
        "--site-tags-root",
        type=Path,
        required=True,
        help="Root directory of deepghs/site_tags clone (contains */tags.sqlite)",
    )
    p.add_argument("--out-dir", type=Path, required=True, help="Output directory for TSV reports")
    args = p.parse_args()

    bad = _fetch_bad_tags(args.db)
    out_dir = Path(args.out_dir)

    # 1) Raw list
    raw_rows: list[list[Any]] = []
    for r in bad:
        raw_rows.append(
            [
                r.format_id,
                r.format_name,
                r.tag_id,
                r.tag,
                r.source_tag,
                r.alias,
                r.preferred_tag_id,
                r.deprecated,
                r.usage_count,
                _to_unicode_escape(r.tag),
                _to_unicode_escape(r.source_tag),
            ]
        )
    _write_tsv(
        out_dir / "bad_unicode_tags.tsv",
        [
            "format_id",
            "format_name",
            "tag_id",
            "tag",
            "source_tag",
            "alias",
            "preferred_tag_id",
            "deprecated",
            "usage_count",
            "tag_unicode_escape",
            "source_tag_unicode_escape",
        ],
        raw_rows,
    )

    # 2) Origins + candidate suggestions
    detailed_rows: list[list[Any]] = []
    for r in bad:
        # Candidate: try replacing U+FFFD
        candidates: list[tuple[str, str]] = [
            ("replace_fffd_with_*", r.tag.replace("\ufffd", "*")),
            ("replace_fffd_with_e", r.tag.replace("\ufffd", "e")),
            ("remove_fffd", r.tag.replace("\ufffd", "")),
        ]
        found: list[tuple[bool, int, str, int, str]] = []
        for idx, (rule, cand) in enumerate(candidates):
            tid = _lookup_tag_id_by_exact_tag(args.db, cand)
            if tid is None:
                continue
            has_status = _has_tag_status(args.db, tid, r.format_id)
            found.append((has_status, idx, rule, tid, cand))

        cand_rule = None
        cand_tag = None
        cand_tag_id = None
        if found:
            # Prefer candidates that already have TAG_STATUS for the same format_id.
            found.sort(key=lambda x: (not x[0], x[1]))
            _, _, cand_rule, cand_tag_id, cand_tag = found[0]

        site_dir = _format_to_site_dir(r.format_name)
        origin = None
        origin_path = None
        if site_dir is not None:
            site_sqlite = args.site_tags_root / site_dir / "tags.sqlite"
            if site_sqlite.exists():
                origin_path = str(site_sqlite)
                origin = _site_tags_find_origin(site_sqlite, r.source_tag)
                if origin is None:
                    origin = _site_tags_find_origin(site_sqlite, r.tag)
                if origin is None and r.preferred_tag_id != r.tag_id:
                    preferred_tag = _lookup_tag_by_id(args.db, r.preferred_tag_id)
                    if preferred_tag is not None:
                        origin = _site_tags_find_alias_by_target_tag(site_sqlite, preferred_tag)

        # Flatten minimal origin fields
        origin_kind = origin.get("_origin") if isinstance(origin, dict) else None
        origin_id = None
        origin_count = None
        origin_type = None
        origin_tag_target = None
        origin_hex = None
        if isinstance(origin, dict):
            if "id" in origin:
                origin_id = origin["id"]
            if "post_count" in origin:
                origin_count = origin["post_count"]
            elif "count" in origin:
                origin_count = origin["count"]
            if "type" in origin:
                origin_type = origin["type"]
            if origin_kind is not None and origin_kind.startswith("tag_aliases"):
                origin_tag_target = origin.get("tag")
                origin_hex = origin.get("_hex_alias")
            else:
                origin_hex = origin.get("_hex")

        detailed_rows.append(
            [
                r.format_id,
                r.format_name,
                r.tag_id,
                r.tag,
                r.source_tag,
                r.alias,
                r.preferred_tag_id,
                r.deprecated,
                r.usage_count,
                cand_rule,
                cand_tag_id,
                cand_tag,
                origin_path,
                origin_kind,
                origin_id,
                origin_type,
                origin_count,
                origin_tag_target,
                origin_hex,
            ]
        )

    _write_tsv(
        out_dir / "bad_unicode_tag_origins_and_candidates.tsv",
        [
            "format_id",
            "format_name",
            "bad_tag_id",
            "bad_tag",
            "bad_source_tag",
            "alias",
            "preferred_tag_id",
            "deprecated",
            "usage_count",
            "suggest_rule",
            "suggest_tag_id",
            "suggest_tag",
            "origin_sqlite",
            "origin_kind",
            "origin_row_id",
            "origin_type",
            "origin_count",
            "origin_target_tag",
            "origin_hex",
        ],
        detailed_rows,
    )

    print(f"Wrote: {out_dir / 'bad_unicode_tags.tsv'}")
    print(f"Wrote: {out_dir / 'bad_unicode_tag_origins_and_candidates.tsv'}")


if __name__ == "__main__":
    main()
