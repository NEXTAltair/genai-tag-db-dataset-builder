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
                # deepghs/site_tags integration (format_name is our internal key, not domain)
                (4, "safebooru", ""),
                (5, "allthefallen", ""),
                (6, "sankaku", ""),
                (7, "gelbooru", ""),
                (8, "rule34", ""),
                (9, "xbooru", ""),
                (10, "hypnohub", ""),
                (11, "konachan", ""),
                (12, "konachan-net", ""),
                (13, "lolibooru", ""),
                (14, "anime-pictures", ""),
                (15, "wallhaven", ""),
                (16, "yandere", ""),
                (17, "zerochan", ""),
                (18, "pixiv", ""),
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
                (17, "contributor", ""),
                (18, "organization", ""),
                (19, "deprecated", ""),
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
            (2, 2, 17, "contributor"),
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

        # deepghs/site_tags: additional formats (best-effort baseline)
        danbooru_like_mappings = [
            (0, 1, "general"),
            (1, 2, "artist"),
            (3, 3, "copyright"),
            (4, 4, "character"),
            (5, 7, "meta"),
        ]
        safebooru_mappings = [(4, t, n, d) for (t, n, d) in danbooru_like_mappings]
        allthefallen_mappings = [(5, t, n, d) for (t, n, d) in danbooru_like_mappings]

        sankaku_mappings = [
            (6, 0, 1, "general"),
            (6, 1, 2, "artist"),
            (6, 2, 18, "organization"),
            (6, 3, 3, "copyright"),
            (6, 4, 4, "character"),
            (6, 8, 7, "meta"),
            (6, 9, 7, "meta"),
        ]

        # gelbooru: source uses string types; adapter should map to these numeric type_id values.
        gelbooru_mappings = [
            (7, 0, 1, "general"),
            (7, 1, 2, "artist"),
            (7, 3, 3, "copyright"),
            (7, 4, 4, "character"),
            (7, 5, 7, "meta"),  # metadata -> meta
            (7, 6, 19, "deprecated"),  # deprecated -> deprecated
        ]

        moebooru_mappings = [
            (0, 1, "general"),
            (1, 2, "artist"),
            (3, 3, "copyright"),
            (4, 4, "character"),
            (5, 7, "meta"),
        ]
        rule34_mappings = [(8, t, n, d) for (t, n, d) in moebooru_mappings]
        xbooru_mappings = [(9, t, n, d) for (t, n, d) in moebooru_mappings]
        hypnohub_mappings = [(10, t, n, d) for (t, n, d) in moebooru_mappings]
        konachan_mappings = [(11, t, n, d) for (t, n, d) in moebooru_mappings] + [
            (11, 6, 18, "organization")
        ]
        konachan_net_mappings = [(12, t, n, d) for (t, n, d) in moebooru_mappings] + [
            (12, 6, 18, "organization")
        ]
        yandere_mappings = [(16, t, n, d) for (t, n, d) in moebooru_mappings] + [(16, 6, 7, "meta")]

        lolibooru_mappings = [
            (13, 0, 1, "general"),
            (13, 1, 2, "artist"),
            (13, 3, 3, "copyright"),
            (13, 4, 4, "character"),
            (13, 5, 18, "organization"),
            (13, 6, 7, "meta"),
        ]

        anime_pictures_mappings = [
            (14, 0, 7, "meta"),
            (14, 1, 4, "character"),
            (14, 2, 1, "general"),
            (14, 3, 3, "copyright"),
            (14, 4, 2, "artist"),
            (14, 5, 3, "copyright"),
            (14, 6, 3, "copyright"),
            (14, 7, 1, "general"),
        ]

        all_mappings = (
            danbooru_mappings
            + e621_mappings
            + derpibooru_mappings
            + safebooru_mappings
            + allthefallen_mappings
            + sankaku_mappings
            + gelbooru_mappings
            + rule34_mappings
            + xbooru_mappings
            + hypnohub_mappings
            + konachan_mappings
            + konachan_net_mappings
            + yandere_mappings
            + lolibooru_mappings
            + anime_pictures_mappings
        )

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
