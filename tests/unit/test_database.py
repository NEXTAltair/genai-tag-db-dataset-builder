"""database.py のユニットテスト（作成・インデックス・最適化）."""

import sqlite3
from pathlib import Path

import pytest

from genai_tag_db_dataset_builder.core.database import (
    BUILD_TIME_PRAGMAS,
    DISTRIBUTION_PRAGMAS,
    REQUIRED_INDEXES,
    build_indexes,
    create_database,
    optimize_database,
)


class TestCreateDatabase:
    """create_database関数のテスト."""

    def test_create_new_database(self, tmp_path: Path) -> None:
        """新規データベース作成のテスト."""
        db_path = tmp_path / "test.db"
        create_database(db_path)

        assert db_path.exists()

        # page_size確認
        conn = sqlite3.connect(db_path)
        cursor = conn.execute("PRAGMA page_size;")
        page_size = cursor.fetchone()[0]
        conn.close()

        assert page_size == 4096

    def test_create_database_already_exists(self, tmp_path: Path) -> None:
        """既存データベースに対する作成のテスト."""
        db_path = tmp_path / "test.db"

        # 先に作成
        create_database(db_path)
        original_size = db_path.stat().st_size

        # 再度作成（警告のみ、エラーにならない）
        create_database(db_path)

        # ファイルサイズが変わらないことを確認
        assert db_path.stat().st_size == original_size

    def test_auto_vacuum_setting(self, tmp_path: Path) -> None:
        """auto_vacuum設定のテスト."""
        db_path = tmp_path / "test.db"
        create_database(db_path)

        conn = sqlite3.connect(db_path)
        cursor = conn.execute("PRAGMA auto_vacuum;")
        auto_vacuum = cursor.fetchone()[0]
        conn.close()

        # INCREMENTAL = 2
        assert auto_vacuum == 2


class TestBuildIndexes:
    """build_indexes関数のテスト."""

    def test_build_indexes_success(self, tmp_path: Path) -> None:
        """インデックス構築のテスト."""
        db_path = tmp_path / "test.db"

        # データベース作成
        conn = sqlite3.connect(db_path)
        # テストテーブル作成（TAGS, TAG_STATUS, TAG_TRANSLATIONS, TAG_USAGE_COUNTS）
        conn.execute(
            """
            CREATE TABLE TAGS (
                tag_id INTEGER PRIMARY KEY,
                tag TEXT NOT NULL UNIQUE,
                source_tag TEXT NOT NULL
            )
        """
        )
        conn.execute(
            """
            CREATE TABLE TAG_STATUS (
                tag_id INTEGER NOT NULL,
                format_id INTEGER NOT NULL,
                type_id INTEGER,
                alias BOOLEAN NOT NULL,
                preferred_tag_id INTEGER NOT NULL,
                PRIMARY KEY (tag_id, format_id)
            )
        """
        )
        conn.execute(
            """
            CREATE TABLE TAG_TRANSLATIONS (
                translation_id INTEGER PRIMARY KEY,
                tag_id INTEGER NOT NULL,
                language TEXT NOT NULL,
                translation TEXT NOT NULL
            )
        """
        )
        conn.execute(
            """
            CREATE TABLE TAG_USAGE_COUNTS (
                tag_id INTEGER NOT NULL,
                format_id INTEGER NOT NULL,
                count INTEGER NOT NULL,
                PRIMARY KEY (tag_id, format_id)
            )
        """
        )
        conn.commit()
        conn.close()

        # インデックス構築
        build_indexes(db_path)

        # インデックスが作成されたことを確認
        conn = sqlite3.connect(db_path)
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%';")
        indexes = [row[0] for row in cursor.fetchall()]
        conn.close()

        assert len(indexes) == len(REQUIRED_INDEXES)
        assert "idx_tags_tag" in indexes
        assert "idx_tags_source_tag" in indexes

    def test_build_indexes_nonexistent_db(self, tmp_path: Path) -> None:
        """存在しないデータベースに対するインデックス構築のテスト."""
        db_path = tmp_path / "nonexistent.db"

        with pytest.raises(FileNotFoundError):
            build_indexes(db_path)


class TestOptimizeDatabase:
    """optimize_database関数のテスト."""

    def test_optimize_database_success(self, tmp_path: Path) -> None:
        """データベース最適化のテスト."""
        db_path = tmp_path / "test.db"

        # データベース作成
        create_database(db_path)

        # 最適化実行
        optimize_database(db_path)

        # journal_modeがWALに変更されたことを確認
        conn = sqlite3.connect(db_path)
        cursor = conn.execute("PRAGMA journal_mode;")
        journal_mode = cursor.fetchone()[0]
        conn.close()

        assert journal_mode == "wal"

    def test_optimize_database_nonexistent_db(self, tmp_path: Path) -> None:
        """存在しないデータベースに対する最適化のテスト."""
        db_path = tmp_path / "nonexistent.db"

        with pytest.raises(FileNotFoundError):
            optimize_database(db_path)


class TestPragmaSettings:
    """PRAGMA設定のテスト."""

    def test_build_time_pragmas_count(self) -> None:
        """ビルド時PRAGMA設定数のテスト."""
        assert len(BUILD_TIME_PRAGMAS) == 5

    def test_distribution_pragmas_count(self) -> None:
        """配布時PRAGMA設定数のテスト."""
        assert len(DISTRIBUTION_PRAGMAS) == 5

    def test_required_indexes_count(self) -> None:
        """必須インデックス数のテスト."""
        assert len(REQUIRED_INDEXES) == 8
