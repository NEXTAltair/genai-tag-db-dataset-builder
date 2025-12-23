"""SQLiteデータベース作成・最適化ユーティリティ.

タグDBビルダー用の SQLite 作成、インデックス作成、最適化（VACUUM/ANALYZE）を提供します。

注意:
    PRAGMA のうち、cache_size / temp_store / mmap_size / locking_mode などは接続単位の設定です。
    DBファイルへ恒久的に「書き込まれる設定」ではない点に注意してください。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from loguru import logger

# PRAGMA は「DBファイルに永続化されるもの」と「接続ごとの一時設定」が混在するため、
# 意図が伝わるように分類して定義する。
PERSISTENT_BUILD_PRAGMAS = [
    "PRAGMA journal_mode = OFF;",  # WALオフ（ビルド高速化）
    "PRAGMA synchronous = OFF;",  # 同期オフ（ビルド高速化）
]
CONNECTION_BUILD_PRAGMAS = [
    "PRAGMA cache_size = -128000;",  # 128MB cache
    "PRAGMA temp_store = MEMORY;",
    "PRAGMA locking_mode = EXCLUSIVE;",  # 排他ロック（単一プロセス）
]

PERSISTENT_DISTRIBUTION_PRAGMAS = [
    "PRAGMA journal_mode = WAL;",  # WAL有効（読み取り並行性）
    "PRAGMA synchronous = NORMAL;",
]
CONNECTION_DISTRIBUTION_PRAGMAS = [
    "PRAGMA cache_size = -64000;",  # 64MB cache
    "PRAGMA temp_store = MEMORY;",
    "PRAGMA mmap_size = 268435456;",  # 256MB mmap
]

# 互換性のため、従来の定数名も維持する（内容は「永続 + 接続」の合成）。
BUILD_TIME_PRAGMAS = [*PERSISTENT_BUILD_PRAGMAS, *CONNECTION_BUILD_PRAGMAS]
DISTRIBUTION_PRAGMAS = [
    *PERSISTENT_DISTRIBUTION_PRAGMAS,
    *CONNECTION_DISTRIBUTION_PRAGMAS,
]


def apply_connection_pragmas(conn: sqlite3.Connection, *, profile: str) -> None:
    """接続ごとに適用が必要な PRAGMA を設定する。"""
    if profile == "build":
        pragmas = CONNECTION_BUILD_PRAGMAS
    elif profile == "distribution":
        pragmas = CONNECTION_DISTRIBUTION_PRAGMAS
    else:
        raise ValueError(f"Unknown PRAGMA profile: {profile!r}")

    for pragma in pragmas:
        conn.execute(pragma)


# 必須インデックス（想定クエリに基づく）
REQUIRED_INDEXES = [
    # TAGS検索用（部分一致・完全一致）
    "CREATE INDEX IF NOT EXISTS idx_tags_tag ON TAGS(tag);",
    "CREATE INDEX IF NOT EXISTS idx_tags_source_tag ON TAGS(source_tag);",
    # TAG_STATUS検索用（format/type/置換先）
    "CREATE INDEX IF NOT EXISTS idx_tag_status_format ON TAG_STATUS(format_id);",
    "CREATE INDEX IF NOT EXISTS idx_tag_status_type ON TAG_STATUS(type_id);",
    "CREATE INDEX IF NOT EXISTS idx_tag_status_preferred ON TAG_STATUS(preferred_tag_id);",
    # TAG_TRANSLATIONS検索用（言語別・翻訳検索）
    "CREATE INDEX IF NOT EXISTS idx_translations_tag_lang ON TAG_TRANSLATIONS(tag_id, language);",
    "CREATE INDEX IF NOT EXISTS idx_translations_text ON TAG_TRANSLATIONS(translation);",
    # TAG_USAGE_COUNTS検索用（count順ソート）
    "CREATE INDEX IF NOT EXISTS idx_usage_counts_count ON TAG_USAGE_COUNTS(count DESC);",
]

# DBスキーマ（tags_v4.db 互換ベース）
SCHEMA_SQL = [
    """
    CREATE TABLE IF NOT EXISTS TAGS (
        tag_id INTEGER NOT NULL PRIMARY KEY,
        source_tag TEXT NOT NULL,
        tag TEXT NOT NULL,
        created_at DATETIME DEFAULT (CURRENT_TIMESTAMP),
        updated_at DATETIME DEFAULT (CURRENT_TIMESTAMP),
        UNIQUE(tag)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS TAG_FORMATS (
        format_id INTEGER NOT NULL PRIMARY KEY,
        format_name TEXT NOT NULL,
        description TEXT,
        UNIQUE(format_name)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS TAG_TYPE_NAME (
        type_name_id INTEGER NOT NULL PRIMARY KEY,
        type_name TEXT NOT NULL,
        description TEXT,
        UNIQUE(type_name)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS TAG_TYPE_FORMAT_MAPPING (
        format_id INTEGER NOT NULL,
        type_id INTEGER NOT NULL,
        type_name_id INTEGER NOT NULL,
        description TEXT,
        PRIMARY KEY (format_id, type_id),
        FOREIGN KEY(format_id) REFERENCES TAG_FORMATS(format_id),
        FOREIGN KEY(type_name_id) REFERENCES TAG_TYPE_NAME(type_name_id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS TAG_STATUS (
        tag_id INTEGER NOT NULL,
        format_id INTEGER NOT NULL,
        type_id INTEGER NOT NULL,
        alias BOOLEAN NOT NULL,
        preferred_tag_id INTEGER NOT NULL,
        deprecated BOOLEAN NOT NULL DEFAULT 0,
        deprecated_at DATETIME NULL,
        source_created_at DATETIME NULL,
        created_at DATETIME DEFAULT (CURRENT_TIMESTAMP),
        updated_at DATETIME DEFAULT (CURRENT_TIMESTAMP),
        PRIMARY KEY (tag_id, format_id),
        FOREIGN KEY(tag_id) REFERENCES TAGS(tag_id),
        FOREIGN KEY(format_id) REFERENCES TAG_FORMATS(format_id),
        FOREIGN KEY(preferred_tag_id) REFERENCES TAGS(tag_id),
        FOREIGN KEY(format_id, type_id) REFERENCES TAG_TYPE_FORMAT_MAPPING(format_id, type_id),
        CONSTRAINT ck_preferred_tag_consistency CHECK (
            (alias = false AND preferred_tag_id = tag_id) OR
            (alias = true AND preferred_tag_id != tag_id)
        )
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS TAG_TRANSLATIONS (
        translation_id INTEGER NOT NULL PRIMARY KEY,
        tag_id INTEGER NOT NULL,
        language TEXT,
        translation TEXT,
        created_at DATETIME DEFAULT (CURRENT_TIMESTAMP),
        updated_at DATETIME DEFAULT (CURRENT_TIMESTAMP),
        FOREIGN KEY(tag_id) REFERENCES TAGS(tag_id),
        UNIQUE(tag_id, language, translation)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS TAG_USAGE_COUNTS (
        tag_id INTEGER NOT NULL,
        format_id INTEGER NOT NULL,
        count INTEGER NOT NULL,
        created_at DATETIME DEFAULT (CURRENT_TIMESTAMP),
        updated_at DATETIME DEFAULT (CURRENT_TIMESTAMP),
        PRIMARY KEY (tag_id, format_id),
        FOREIGN KEY(tag_id) REFERENCES TAGS(tag_id),
        FOREIGN KEY(format_id) REFERENCES TAG_FORMATS(format_id),
        UNIQUE(tag_id, format_id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS DATABASE_METADATA (
        key TEXT NOT NULL PRIMARY KEY,
        value TEXT NOT NULL
    );
    """,
]


def create_schema(db_path: Path | str) -> None:
    """DBスキーマ（テーブル）を作成する."""
    db_path = Path(db_path)
    if not db_path.exists():
        msg = f"Database does not exist: {db_path}"
        raise FileNotFoundError(msg)

    conn = sqlite3.connect(db_path)
    try:
        for stmt in SCHEMA_SQL:
            conn.executescript(stmt)
        conn.commit()
    finally:
        conn.close()


def create_database(db_path: Path | str) -> None:
    """データベースファイルを新規作成する（page_size / auto_vacuum 設定込み）.

    Args:
        db_path: 作成するデータベースファイルパス

    Note:
        page_size と auto_vacuum はDB作成前にのみ有効です。
    """
    db_path = Path(db_path)

    if db_path.exists():
        logger.warning(f"Database already exists: {db_path}")
        return

    db_path.parent.mkdir(parents=True, exist_ok=True)
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

        # スキーマ作成（初回のみ）
        for stmt in SCHEMA_SQL:
            conn.executescript(stmt)

        conn.commit()
        logger.info("Database created successfully")

    except Exception as e:
        logger.error(f"Failed to create database: {e}")
        raise
    finally:
        conn.close()


def build_indexes(db_path: Path | str) -> None:
    """必須インデックスを作成する.

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
    """データベースを最適化する（ビルド完了後の処理）.

    Args:
        db_path: データベースファイルパス

    Note:
        正しい順序は VACUUM → ANALYZE です。
        - VACUUM: 断片化解消、削除領域回収、インデックス再構築
        - ANALYZE: インデックス統計の収集（VACUUM後に必要）
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

        # 3. 配布時PRAGMA設定
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
