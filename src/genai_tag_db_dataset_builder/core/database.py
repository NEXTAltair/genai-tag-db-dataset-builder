"""Database creation and optimization utilities.

This module provides SQLite database creation and optimization functionality
for the tag database builder.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from loguru import logger

# ビルド時PRAGMA設定（速度優先）
BUILD_TIME_PRAGMAS = [
    "PRAGMA journal_mode = OFF;",  # WALオフ（ビルド高速化）
    "PRAGMA synchronous = OFF;",  # 同期オフ（ビルド高速化）
    "PRAGMA cache_size = -128000;",  # 128MB cache
    "PRAGMA temp_store = MEMORY;",
    "PRAGMA locking_mode = EXCLUSIVE;",  # 排他ロック（単一プロセス）
]

# 配布時PRAGMA設定（安全性・並行性優先）
DISTRIBUTION_PRAGMAS = [
    "PRAGMA journal_mode = WAL;",  # WAL有効
    "PRAGMA synchronous = NORMAL;",
    "PRAGMA cache_size = -64000;",  # 64MB cache
    "PRAGMA temp_store = MEMORY;",
    "PRAGMA mmap_size = 268435456;",  # 256MB mmap
]

# 必須インデックス（実クエリベース）
REQUIRED_INDEXES = [
    # TAGS検索用（部分一致・完全一致）
    "CREATE INDEX IF NOT EXISTS idx_tags_tag ON TAGS(tag);",
    "CREATE INDEX IF NOT EXISTS idx_tags_source_tag ON TAGS(source_tag);",
    # TAG_STATUS検索用（format別、type別）
    "CREATE INDEX IF NOT EXISTS idx_tag_status_format ON TAG_STATUS(format_id);",
    "CREATE INDEX IF NOT EXISTS idx_tag_status_type ON TAG_STATUS(type_id);",
    "CREATE INDEX IF NOT EXISTS idx_tag_status_preferred ON TAG_STATUS(preferred_tag_id);",
    # TAG_TRANSLATIONS検索用（言語別、翻訳文検索）
    "CREATE INDEX IF NOT EXISTS idx_translations_tag_lang ON TAG_TRANSLATIONS(tag_id, language);",
    "CREATE INDEX IF NOT EXISTS idx_translations_text ON TAG_TRANSLATIONS(translation);",
    # TAG_USAGE_COUNTS検索用（ソート用インデックス）
    "CREATE INDEX IF NOT EXISTS idx_usage_counts_count ON TAG_USAGE_COUNTS(count DESC);",
]


def create_database(db_path: Path | str) -> None:
    """データベース作成（page_size/auto_vacuum設定込み）.

    Args:
        db_path: 作成するデータベースファイルパス

    Note:
        page_sizeとauto_vacuumはDB作成前にのみ設定可能
    """
    db_path = Path(db_path)

    if db_path.exists():
        logger.warning(f"Database already exists: {db_path}")
        return

    logger.info(f"Creating database: {db_path}")

    conn = sqlite3.connect(db_path)

    try:
        # DB作成時にのみ有効な設定
        conn.execute("PRAGMA page_size = 4096;")
        conn.execute("PRAGMA auto_vacuum = INCREMENTAL;")

        # その他のビルド時設定
        for pragma in BUILD_TIME_PRAGMAS:
            conn.execute(pragma)
            logger.debug(f"Applied: {pragma}")

        conn.commit()
        logger.info("Database created successfully")

    except Exception as e:
        logger.error(f"Failed to create database: {e}")
        raise
    finally:
        conn.close()


def build_indexes(db_path: Path | str) -> None:
    """インデックス構築.

    Args:
        db_path: データベースファイルパス
    """
    db_path = Path(db_path)

    if not db_path.exists():
        msg = f"Database does not exist: {db_path}"
        raise FileNotFoundError(msg)

    logger.info(f"Building indexes: {db_path}")

    conn = sqlite3.connect(db_path)

    try:
        for index_sql in REQUIRED_INDEXES:
            logger.debug(f"Creating index: {index_sql}")
            conn.execute(index_sql)

        conn.commit()
        logger.info(f"Created {len(REQUIRED_INDEXES)} indexes successfully")

    except Exception as e:
        logger.error(f"Failed to build indexes: {e}")
        raise
    finally:
        conn.close()


def optimize_database(db_path: Path | str) -> None:
    """データベース最適化（ビルド完了後）.

    Args:
        db_path: データベースファイルパス

    Note:
        正しい順序: VACUUM → ANALYZE
        - VACUUM: 断片化解消、削除された領域回収、インデックス再構築
        - ANALYZE: インデックス統計を収集（VACUUMで再構築されたインデックスの統計が必要）
    """
    db_path = Path(db_path)

    if not db_path.exists():
        msg = f"Database does not exist: {db_path}"
        raise FileNotFoundError(msg)

    logger.info(f"Optimizing database: {db_path}")

    conn = sqlite3.connect(db_path)

    try:
        # 1. VACUUMで断片化解消
        logger.info("Running VACUUM...")
        conn.execute("VACUUM;")

        # 2. ANALYZEでインデックス統計更新
        logger.info("Running ANALYZE...")
        conn.execute("ANALYZE;")

        # 3. 配布用PRAGMA設定
        logger.info("Applying distribution PRAGMA settings...")
        for pragma in DISTRIBUTION_PRAGMAS:
            conn.execute(pragma)
            logger.debug(f"Applied: {pragma}")

        conn.commit()
        logger.info("Optimization complete")

    except Exception as e:
        logger.error(f"Failed to optimize database: {e}")
        raise
    finally:
        conn.close()
