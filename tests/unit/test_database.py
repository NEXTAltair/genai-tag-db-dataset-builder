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

        create_database(db_path)

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


class TestSchemaCompatibility:
    """create_database() のスキーマが期待どおりかを確認する。"""

    def test_schema_matches_expected_columns(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        create_database(db_path)

        # (name, type, not_null, pk)
        expected: dict[str, list[tuple[str, str, bool, bool]]] = {
            "TAGS": [
                ("tag_id", "INTEGER", True, True),
                ("source_tag", "TEXT", True, False),
                ("tag", "TEXT", True, False),
                ("created_at", "DATETIME", False, False),
                ("updated_at", "DATETIME", False, False),
            ],
            "TAG_FORMATS": [
                ("format_id", "INTEGER", True, True),
                ("format_name", "TEXT", True, False),
                ("description", "TEXT", False, False),
            ],
            "TAG_TYPE_NAME": [
                ("type_name_id", "INTEGER", True, True),
                ("type_name", "TEXT", True, False),
                ("description", "TEXT", False, False),
            ],
            "TAG_TYPE_FORMAT_MAPPING": [
                ("format_id", "INTEGER", True, True),
                ("type_id", "INTEGER", True, True),
                ("type_name_id", "INTEGER", True, False),
                ("description", "TEXT", False, False),
            ],
            "TAG_STATUS": [
                ("tag_id", "INTEGER", True, True),
                ("format_id", "INTEGER", True, True),
                ("type_id", "INTEGER", True, False),
                ("alias", "BOOLEAN", True, False),
                ("preferred_tag_id", "INTEGER", True, False),
                ("created_at", "DATETIME", False, False),
                ("updated_at", "DATETIME", False, False),
            ],
            "TAG_TRANSLATIONS": [
                ("translation_id", "INTEGER", True, True),
                ("tag_id", "INTEGER", True, False),
                ("language", "TEXT", False, False),
                ("translation", "TEXT", False, False),
                ("created_at", "DATETIME", False, False),
                ("updated_at", "DATETIME", False, False),
            ],
            "TAG_USAGE_COUNTS": [
                ("tag_id", "INTEGER", True, True),
                ("format_id", "INTEGER", True, True),
                ("count", "INTEGER", True, False),
                ("created_at", "DATETIME", False, False),
                ("updated_at", "DATETIME", False, False),
            ],
        }

        conn = sqlite3.connect(db_path)
        try:
            for table, expected_columns in expected.items():
                rows = conn.execute(f"PRAGMA table_info({table});").fetchall()
                got = [
                    (name, col_type.upper(), bool(notnull), bool(pk))
                    for _, name, col_type, notnull, _, pk in rows
                ]
                assert got == expected_columns
        finally:
            conn.close()
