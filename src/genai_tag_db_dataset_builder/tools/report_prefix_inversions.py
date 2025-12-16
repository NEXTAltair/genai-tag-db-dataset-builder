"""DB内の prefix 逆転（preferred が meta:/artist: 側）を抽出してTSV出力する。"""

from __future__ import annotations

import argparse
import csv
import sqlite3
from pathlib import Path


def write_prefix_preferred_inversion_tsv(db_path: Path, out_path: Path) -> int:
    """preferred_tag が meta:/artist: になっている alias 行を抽出する。

    Returns:
        抽出した行数
    """
    db_path = Path(db_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        rows = cur.execute(
            """
            SELECT
                ts.format_id,
                f.format_name,
                ts.tag_id,
                t.tag AS tag,
                ts.preferred_tag_id,
                tp.tag AS preferred_tag,
                ts.type_id,
                ts.alias
            FROM TAG_STATUS ts
            JOIN TAGS t ON t.tag_id = ts.tag_id
            JOIN TAGS tp ON tp.tag_id = ts.preferred_tag_id
            JOIN TAG_FORMATS f ON f.format_id = ts.format_id
            WHERE
                ts.alias = 1
                AND (
                    tp.tag LIKE 'meta:%'
                    OR tp.tag LIKE 'artist:%'
                )
                AND NOT (
                    t.tag LIKE 'meta:%'
                    OR t.tag LIKE 'artist:%'
                )
            ORDER BY ts.format_id, t.tag
            """
        ).fetchall()

        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(
                [
                    "format_id",
                    "format_name",
                    "tag_id",
                    "tag",
                    "preferred_tag_id",
                    "preferred_tag",
                    "type_id",
                    "alias",
                ]
            )
            for r in rows:
                writer.writerow(
                    [
                        r["format_id"],
                        r["format_name"],
                        r["tag_id"],
                        r["tag"],
                        r["preferred_tag_id"],
                        r["preferred_tag"],
                        r["type_id"],
                        r["alias"],
                    ]
                )

        return len(rows)
    finally:
        con.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export prefix preferred inversion rows to TSV")
    parser.add_argument("--db", type=Path, required=True, help="Path to unified SQLite DB")
    parser.add_argument("--out", type=Path, required=True, help="Output TSV path")
    args = parser.parse_args()

    count = write_prefix_preferred_inversion_tsv(args.db, args.out)
    print(f"Wrote {count} rows: {args.out}")


if __name__ == "__main__":
    main()

