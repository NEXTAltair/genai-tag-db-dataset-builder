"""Unit tests for master data initialization."""

import sqlite3
from pathlib import Path

import pytest

from genai_tag_db_dataset_builder.core.master_data import initialize_master_data


class TestInitializeMasterData:
    """initialize_master_data関数のテスト."""

    def test_initialize_master_data_success(self, tmp_path: Path) -> None:
        """マスタデータ初期化のテスト."""
        db_path = tmp_path / "test.db"

        # データベース作成
        conn = sqlite3.connect(db_path)

        # テーブル作成
        conn.execute(
            """
            CREATE TABLE TAG_FORMATS (
                format_id INTEGER PRIMARY KEY,
                format_name TEXT UNIQUE NOT NULL,
                description TEXT
            )
        """
        )

        conn.execute(
            """
            CREATE TABLE TAG_TYPE_NAME (
                type_name_id INTEGER PRIMARY KEY,
                type_name TEXT UNIQUE NOT NULL,
                description TEXT
            )
        """
        )

        conn.execute(
            """
            CREATE TABLE TAG_TYPE_FORMAT_MAPPING (
                format_id INTEGER NOT NULL,
                type_id INTEGER NOT NULL,
                type_name_id INTEGER NOT NULL,
                description TEXT,
                PRIMARY KEY (format_id, type_id)
            )
        """
        )

        conn.commit()
        conn.close()

        # マスタデータ初期化
        initialize_master_data(db_path)

        # 初期化確認
        conn = sqlite3.connect(db_path)

        # TAG_FORMATS確認（4レコード）
        cursor = conn.execute("SELECT COUNT(*) FROM TAG_FORMATS")
        assert cursor.fetchone()[0] == 19

        # TAG_TYPE_NAME確認（17レコード）
        cursor = conn.execute("SELECT COUNT(*) FROM TAG_TYPE_NAME")
        assert cursor.fetchone()[0] == 20

        # TAG_TYPE_FORMAT_MAPPING確認（25レコード: Danbooru 5 + E621 8 + Derpibooru 12）
        cursor = conn.execute("SELECT COUNT(*) FROM TAG_TYPE_FORMAT_MAPPING")
        assert cursor.fetchone()[0] == 96

        # 具体的なデータ確認
        cursor = conn.execute("SELECT format_name FROM TAG_FORMATS WHERE format_id = 1")
        assert cursor.fetchone()[0] == "danbooru"

        cursor = conn.execute("SELECT type_name FROM TAG_TYPE_NAME WHERE type_name_id = 4")
        assert cursor.fetchone()[0] == "character"

        conn.close()

    def test_initialize_master_data_nonexistent_db(self, tmp_path: Path) -> None:
        """存在しないデータベースに対する初期化のテスト."""
        db_path = tmp_path / "nonexistent.db"

        with pytest.raises(FileNotFoundError):
            initialize_master_data(db_path)

    def test_initialize_master_data_idempotent(self, tmp_path: Path) -> None:
        """マスタデータ初期化の冪等性テスト."""
        db_path = tmp_path / "test.db"

        # データベース作成
        conn = sqlite3.connect(db_path)

        # テーブル作成
        conn.execute(
            """
            CREATE TABLE TAG_FORMATS (
                format_id INTEGER PRIMARY KEY,
                format_name TEXT UNIQUE NOT NULL,
                description TEXT
            )
        """
        )

        conn.execute(
            """
            CREATE TABLE TAG_TYPE_NAME (
                type_name_id INTEGER PRIMARY KEY,
                type_name TEXT UNIQUE NOT NULL,
                description TEXT
            )
        """
        )

        conn.execute(
            """
            CREATE TABLE TAG_TYPE_FORMAT_MAPPING (
                format_id INTEGER NOT NULL,
                type_id INTEGER NOT NULL,
                type_name_id INTEGER NOT NULL,
                description TEXT,
                PRIMARY KEY (format_id, type_id)
            )
        """
        )

        conn.commit()
        conn.close()

        # 1回目の初期化
        initialize_master_data(db_path)

        # 2回目の初期化（エラーにならないこと）
        initialize_master_data(db_path)

        # レコード数が変わっていないことを確認
        conn = sqlite3.connect(db_path)

        cursor = conn.execute("SELECT COUNT(*) FROM TAG_FORMATS")
        assert cursor.fetchone()[0] == 19

        cursor = conn.execute("SELECT COUNT(*) FROM TAG_TYPE_NAME")
        assert cursor.fetchone()[0] == 20

        cursor = conn.execute("SELECT COUNT(*) FROM TAG_TYPE_FORMAT_MAPPING")
        assert cursor.fetchone()[0] == 96

        conn.close()
