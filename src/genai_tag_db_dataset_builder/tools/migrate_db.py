from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from loguru import logger

from genai_tag_db_dataset_builder.core.master_data import initialize_master_data


def _get_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table});").fetchall()
    return {r[1] for r in rows}


def _add_column_if_missing(
    conn: sqlite3.Connection,
    *,
    table: str,
    column: str,
    ddl: str,
) -> bool:
    cols = _get_columns(conn, table)
    if column in cols:
        logger.info(f"Skip (exists): {table}.{column}")
        return False
    logger.info(f"Apply: {ddl}")
    conn.execute(ddl)
    return True


def migrate(db_path: Path) -> int:
    if not db_path.exists():
        raise FileNotFoundError(db_path)

    logger.info(f"Migrating DB: {db_path}")
    changed = 0

    conn = sqlite3.connect(db_path)
    try:
        changed += int(
            _add_column_if_missing(
                conn,
                table="TAG_STATUS",
                column="deprecated",
                ddl="ALTER TABLE TAG_STATUS ADD COLUMN deprecated BOOLEAN NOT NULL DEFAULT 0;",
            )
        )
        changed += int(
            _add_column_if_missing(
                conn,
                table="TAG_STATUS",
                column="deprecated_at",
                ddl="ALTER TABLE TAG_STATUS ADD COLUMN deprecated_at DATETIME NULL;",
            )
        )
        changed += int(
            _add_column_if_missing(
                conn,
                table="TAG_STATUS",
                column="source_created_at",
                ddl="ALTER TABLE TAG_STATUS ADD COLUMN source_created_at DATETIME NULL;",
            )
        )

        conn.commit()
    finally:
        conn.close()

    # Master data is inserted with INSERT OR IGNORE, so it is safe (and desirable)
    # to re-run after schema migrations.
    initialize_master_data(db_path)

    logger.info(f"Migration complete. changed={changed}")
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply schema migrations to an existing genai-image-tag-db SQLite file."
    )
    parser.add_argument("--db", type=Path, required=True, help="SQLite DB path")
    args = parser.parse_args()

    migrate(args.db)


if __name__ == "__main__":
    main()
