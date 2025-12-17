"""マスタデータ初期化.

TAG_FORMATS / TAG_TYPE_NAME / TAG_TYPE_FORMAT_MAPPING の初期データを投入します。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from loguru import logger


def initialize_master_data(db_path: Path | str) -> None:
    """マスタデータを初期化する.

    Args:
        db_path: データベースファイルパス

    Note:
        初回のみ、以下のマスタテーブルを初期化します。
        - TAG_FORMATS（4レコード）
        - TAG_TYPE_NAME（17レコード）
        - TAG_TYPE_FORMAT_MAPPING（25レコード: Danbooru 5 + E621 8 + Derpibooru 12）
    """
    db_path = Path(db_path)

    if not db_path.exists():
        msg = f"Database does not exist: {db_path}"
        raise FileNotFoundError(msg)

    logger.info(f"Initializing master data: {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        # TAG_FORMATS 初期化
        logger.info("Initializing TAG_FORMATS...")
        conn.executemany(
            "INSERT OR IGNORE INTO TAG_FORMATS (format_id, format_name, description) VALUES (?, ?, ?)",
            [
                (0, "unknown", ""),
                (1, "danbooru", ""),
                (2, "e621", ""),
                (3, "derpibooru", ""),
            ],
        )

        # TAG_TYPE_NAME 初期化
        logger.info("Initializing TAG_TYPE_NAME...")
        conn.executemany(
            "INSERT OR IGNORE INTO TAG_TYPE_NAME (type_name_id, type_name, description) VALUES (?, ?, ?)",
            [
                (0, "unknown", ""),
                (1, "general", ""),
                (2, "artist", ""),
                (3, "copyright", ""),
                (4, "character", ""),
                (5, "species", ""),
                (6, "invalid", ""),
                (7, "meta", ""),
                (8, "lore", ""),
                (9, "oc", ""),
                (10, "rating", ""),
                (11, "body-type", ""),
                (12, "origin", ""),
                (13, "error", ""),
                (14, "spoiler", ""),
                (15, "content-official", ""),
                (16, "content-fanmade", ""),
            ],
        )

        # TAG_TYPE_FORMAT_MAPPING 初期化
        logger.info("Initializing TAG_TYPE_FORMAT_MAPPING...")

        # Danbooru (format_id=1)
        danbooru_mappings = [
            (1, 0, 1, "general"),
            (1, 1, 2, "artist"),
            (1, 3, 3, "copyright"),
            (1, 4, 4, "character"),
            (1, 5, 7, "meta"),
        ]

        # E621 (format_id=2)
        e621_mappings = [
            (2, 0, 1, "general"),
            (2, 1, 2, "artist"),
            (2, 3, 3, "copyright"),
            (2, 4, 4, "character"),
            (2, 5, 5, "species"),
            (2, 6, 6, "invalid"),
            (2, 7, 7, "meta"),
            (2, 8, 8, "lore"),
        ]

        # Derpibooru (format_id=3)
        derpibooru_mappings = [
            (3, 0, 1, "general"),
            (3, 1, 15, "content-official"),
            (3, 2, 1, "general"),
            (3, 3, 5, "species"),
            (3, 4, 9, "oc"),
            (3, 5, 10, "rating"),
            (3, 6, 11, "body-type"),
            (3, 7, 7, "meta"),
            (3, 8, 12, "origin"),
            (3, 9, 13, "error"),
            (3, 10, 14, "spoiler"),
            (3, 11, 16, "content-fanmade"),
        ]

        all_mappings = danbooru_mappings + e621_mappings + derpibooru_mappings

        conn.executemany(
            "INSERT OR IGNORE INTO TAG_TYPE_FORMAT_MAPPING (format_id, type_id, type_name_id, description) VALUES (?, ?, ?, ?)",
            all_mappings,
        )

        conn.commit()
        logger.info("Master data initialization complete")

    except Exception as e:
        logger.error(f"Failed to initialize master data: {e}")
        raise
    finally:
        conn.close()
