"""配布用SQLiteの健全性チェックを行い、TSVレポートを出力する。"""

from __future__ import annotations

import argparse
import csv
import sqlite3
from collections.abc import Iterable, Sequence
from pathlib import Path


def _write_tsv(path: Path, header: Sequence[str], rows: Iterable[Sequence[object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(list(header))
        count = 0
        for r in rows:
            writer.writerow(["" if v is None else v for v in r])
            count += 1
    return count


def _fetchall(con: sqlite3.Connection, sql: str, params: Sequence[object] = ()) -> list[sqlite3.Row]:
    cur = con.execute(sql, params)
    return cur.fetchall()


def run_health_checks(db_path: Path, out_dir: Path) -> Path:
    db_path = Path(db_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(db_path)
    try:
        con.row_factory = sqlite3.Row

        # Basic info
        quick_check_rows = _fetchall(con, "PRAGMA quick_check;")
        quick_check = "|".join([r[0] for r in quick_check_rows]) if quick_check_rows else ""

        totals = {
            "tags": _fetchall(con, "SELECT COUNT(*) AS n FROM TAGS;")[0]["n"],
            "tag_status": _fetchall(con, "SELECT COUNT(*) AS n FROM TAG_STATUS;")[0]["n"],
            "translations": _fetchall(con, "SELECT COUNT(*) AS n FROM TAG_TRANSLATIONS;")[0]["n"],
            "usage_counts": _fetchall(con, "SELECT COUNT(*) AS n FROM TAG_USAGE_COUNTS;")[0]["n"],
        }

        # Foreign key check (if schema defines FK; empty = OK)
        fk_rows = _fetchall(con, "PRAGMA foreign_key_check;")
        fk_out = out_dir / "foreign_key_check.tsv"
        fk_count = _write_tsv(
            fk_out,
            ["table", "rowid", "parent", "fkid"],
            [(r["table"], r["rowid"], r["parent"], r["fkid"]) for r in fk_rows],
        )

        # FK違反の大半が TAG_STATUS -> TAG_TYPE_FORMAT_MAPPING なので、原因を集計しておく
        missing_type_format = _fetchall(
            con,
            """
            SELECT
                ts.format_id,
                f.format_name,
                ts.type_id,
                COUNT(*) AS rows
            FROM TAG_STATUS ts
            JOIN TAG_FORMATS f ON f.format_id = ts.format_id
            LEFT JOIN TAG_TYPE_FORMAT_MAPPING m
                ON m.format_id = ts.format_id
                AND m.type_id = ts.type_id
            WHERE m.format_id IS NULL
            GROUP BY ts.format_id, f.format_name, ts.type_id
            ORDER BY rows DESC, ts.format_id, ts.type_id
            """,
        )
        missing_type_format_out = out_dir / "missing_type_format_mapping.tsv"
        missing_type_format_count = _write_tsv(
            missing_type_format_out,
            ["format_id", "format_name", "type_id", "rows"],
            [(r["format_id"], r["format_name"], r["type_id"], r["rows"]) for r in missing_type_format],
        )

        # Orphans
        orphan_tag_status_tag = _fetchall(
            con,
            """
            SELECT ts.format_id, ts.tag_id
            FROM TAG_STATUS ts
            LEFT JOIN TAGS t ON t.tag_id = ts.tag_id
            WHERE t.tag_id IS NULL
            ORDER BY ts.format_id, ts.tag_id
            """,
        )
        orphan_tag_status_pref = _fetchall(
            con,
            """
            SELECT ts.format_id, ts.preferred_tag_id
            FROM TAG_STATUS ts
            LEFT JOIN TAGS t ON t.tag_id = ts.preferred_tag_id
            WHERE t.tag_id IS NULL
            ORDER BY ts.format_id, ts.preferred_tag_id
            """,
        )
        orphan_usage = _fetchall(
            con,
            """
            SELECT uc.format_id, uc.tag_id, uc.count
            FROM TAG_USAGE_COUNTS uc
            LEFT JOIN TAGS t ON t.tag_id = uc.tag_id
            WHERE t.tag_id IS NULL
            ORDER BY uc.format_id, uc.tag_id
            """,
        )
        orphan_trans = _fetchall(
            con,
            """
            SELECT tr.tag_id, tr.language, tr.translation
            FROM TAG_TRANSLATIONS tr
            LEFT JOIN TAGS t ON t.tag_id = tr.tag_id
            WHERE t.tag_id IS NULL
            ORDER BY tr.tag_id, tr.language
            """,
        )

        orphan_tag_status_out = out_dir / "orphan_tag_status.tsv"
        orphan_tag_status_count = _write_tsv(
            orphan_tag_status_out,
            ["kind", "format_id", "id"],
            [("tag_id", r["format_id"], r["tag_id"]) for r in orphan_tag_status_tag]
            + [("preferred_tag_id", r["format_id"], r["preferred_tag_id"]) for r in orphan_tag_status_pref],
        )

        orphan_usage_out = out_dir / "orphan_usage_counts.tsv"
        orphan_usage_count = _write_tsv(
            orphan_usage_out,
            ["format_id", "tag_id", "count"],
            [(r["format_id"], r["tag_id"], r["count"]) for r in orphan_usage],
        )

        orphan_trans_out = out_dir / "orphan_translations.tsv"
        orphan_trans_count = _write_tsv(
            orphan_trans_out,
            ["tag_id", "language", "translation"],
            [(r["tag_id"], r["language"], r["translation"]) for r in orphan_trans],
        )

        # Duplicates
        dup_tags = _fetchall(
            con,
            """
            SELECT tag, COUNT(*) AS n
            FROM TAGS
            GROUP BY tag
            HAVING n > 1
            ORDER BY n DESC, tag
            """,
        )
        dup_tags_out = out_dir / "duplicate_tags.tsv"
        dup_tags_count = _write_tsv(
            dup_tags_out,
            ["tag", "count"],
            [(r["tag"], r["n"]) for r in dup_tags],
        )

        dup_status = _fetchall(
            con,
            """
            SELECT tag_id, format_id, COUNT(*) AS n
            FROM TAG_STATUS
            GROUP BY tag_id, format_id
            HAVING n > 1
            ORDER BY n DESC, format_id, tag_id
            """,
        )
        dup_status_out = out_dir / "duplicate_tag_status.tsv"
        dup_status_count = _write_tsv(
            dup_status_out,
            ["tag_id", "format_id", "count"],
            [(r["tag_id"], r["format_id"], r["n"]) for r in dup_status],
        )

        # Alias consistency (alias=0 should always self-point)
        alias_incons = _fetchall(
            con,
            """
            SELECT ts.format_id, ts.tag_id, t.tag AS tag, ts.alias, ts.preferred_tag_id, tp.tag AS preferred_tag
            FROM TAG_STATUS ts
            JOIN TAGS t ON t.tag_id = ts.tag_id
            JOIN TAGS tp ON tp.tag_id = ts.preferred_tag_id
            WHERE ts.alias = 0 AND ts.preferred_tag_id != ts.tag_id
            ORDER BY ts.format_id, t.tag
            """,
        )
        alias_incons_out = out_dir / "alias_inconsistencies.tsv"
        alias_incons_count = _write_tsv(
            alias_incons_out,
            ["format_id", "tag_id", "tag", "alias", "preferred_tag_id", "preferred_tag"],
            [
                (
                    r["format_id"],
                    r["tag_id"],
                    r["tag"],
                    r["alias"],
                    r["preferred_tag_id"],
                    r["preferred_tag"],
                )
                for r in alias_incons
            ],
        )

        # Usage count sanity
        bad_counts = _fetchall(
            con,
            """
            SELECT uc.format_id, uc.tag_id, t.tag AS tag, uc.count
            FROM TAG_USAGE_COUNTS uc
            JOIN TAGS t ON t.tag_id = uc.tag_id
            WHERE uc.count IS NULL OR uc.count < 0
            ORDER BY uc.format_id, uc.count, t.tag
            """,
        )
        bad_counts_out = out_dir / "bad_usage_counts.tsv"
        bad_counts_count = _write_tsv(
            bad_counts_out,
            ["format_id", "tag_id", "tag", "count"],
            [(r["format_id"], r["tag_id"], r["tag"], r["count"]) for r in bad_counts],
        )

        summary_out = out_dir / "db_health_summary.tsv"
        _write_tsv(
            summary_out,
            ["metric", "value"],
            [
                ("db_path", str(db_path)),
                ("quick_check", quick_check),
                ("total_tags", totals["tags"]),
                ("total_tag_status", totals["tag_status"]),
                ("total_translations", totals["translations"]),
                ("total_usage_counts", totals["usage_counts"]),
                ("foreign_key_violations", fk_count),
                ("missing_type_format_mapping_pairs", missing_type_format_count),
                ("orphan_tag_status", orphan_tag_status_count),
                ("orphan_usage_counts", orphan_usage_count),
                ("orphan_translations", orphan_trans_count),
                ("duplicate_tags", dup_tags_count),
                ("duplicate_tag_status", dup_status_count),
                ("alias_inconsistencies", alias_incons_count),
                ("bad_usage_counts", bad_counts_count),
            ],
        )

        return summary_out
    finally:
        con.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Check SQLite DB health and write TSV reports.")
    p.add_argument("--db", type=Path, required=True, help="Path to SQLite DB file")
    p.add_argument("--out-dir", type=Path, required=True, help="Output directory for TSV reports")
    args = p.parse_args()

    summary = run_health_checks(args.db, args.out_dir)
    print(f"Wrote health reports: {summary.parent}")


if __name__ == "__main__":
    main()
