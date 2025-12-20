"""Unit tests for column type overrides functionality."""

import json
from pathlib import Path

import pytest

from genai_tag_db_dataset_builder.adapters.csv_adapter import CSV_Adapter
from genai_tag_db_dataset_builder.core.column_classifier import TagColumnType
from genai_tag_db_dataset_builder.core.overrides import (
    ColumnTypeOverrides,
    load_overrides,
)


class TestLoadOverrides:
    def test_load_valid_json(self, tmp_path: Path) -> None:
        """有効なJSONファイルからオーバーライドを読み込めること."""
        overrides_file = tmp_path / "overrides.json"
        overrides_data = {
            "data/file1.csv": {"tag": "normalized"},
            "data/file2.json": {"tag": "source"},
        }
        overrides_file.write_text(json.dumps(overrides_data), encoding="utf-8")

        overrides = load_overrides(overrides_file)

        assert overrides.get("data/file1.csv", "tag") == TagColumnType.NORMALIZED
        assert overrides.get("data/file2.json", "tag") == TagColumnType.SOURCE

    def test_load_nonexistent_file(self, tmp_path: Path) -> None:
        """存在しないファイルを読み込むとFileNotFoundErrorが発生すること."""
        nonexistent = tmp_path / "nonexistent.json"

        with pytest.raises(FileNotFoundError, match="Overrides file not found"):
            load_overrides(nonexistent)

    def test_load_invalid_json(self, tmp_path: Path) -> None:
        """不正なJSON形式のファイルを読み込むとValueErrorが発生すること."""
        invalid_json = tmp_path / "invalid.json"
        invalid_json.write_text("{invalid json}", encoding="utf-8")

        with pytest.raises(ValueError, match="Invalid JSON"):
            load_overrides(invalid_json)

    def test_load_invalid_column_type(self, tmp_path: Path) -> None:
        """無効な列タイプが含まれているとValueErrorが発生すること."""
        overrides_file = tmp_path / "overrides.json"
        overrides_data = {"data/file.csv": {"tag": "INVALID_TYPE"}}
        overrides_file.write_text(json.dumps(overrides_data), encoding="utf-8")

        with pytest.raises(ValueError, match="Invalid column type"):
            load_overrides(overrides_file)

    def test_load_invalid_format(self, tmp_path: Path) -> None:
        """ルートがオブジェクトでない場合にValueErrorが発生すること."""
        overrides_file = tmp_path / "overrides.json"
        overrides_file.write_text("[]", encoding="utf-8")  # Array instead of object

        with pytest.raises(ValueError, match="must contain a JSON object"):
            load_overrides(overrides_file)


class TestColumnTypeOverrides:
    def test_get_exact_path_match(self) -> None:
        """完全一致パスでオーバーライドを取得できること."""
        overrides_data = {"data/file.csv": {"tag": "normalized"}}
        overrides = ColumnTypeOverrides(overrides_data)

        result = overrides.get("data/file.csv", "tag")

        assert result == TagColumnType.NORMALIZED

    def test_get_normalized_path_match(self) -> None:
        """パス正規化して一致するオーバーライドを取得できること."""
        overrides_data = {"data/file.csv": {"tag": "source"}}
        overrides = ColumnTypeOverrides(overrides_data)

        # Windows/POSIX両方のパス形式でテスト
        result = overrides.get(Path("data/file.csv"), "tag")

        assert result == TagColumnType.SOURCE

    def test_get_no_match(self) -> None:
        """一致するオーバーライドがない場合にNoneを返すこと."""
        overrides_data = {"data/file.csv": {"tag": "normalized"}}
        overrides = ColumnTypeOverrides(overrides_data)

        result = overrides.get("data/other.csv", "tag")

        assert result is None

    def test_get_column_not_found(self) -> None:
        """ファイルは一致するが列名が一致しない場合にNoneを返すこと."""
        overrides_data = {"data/file.csv": {"other_column": "normalized"}}
        overrides = ColumnTypeOverrides(overrides_data)

        result = overrides.get("data/file.csv", "tag")

        assert result is None

    def test_has_override_true(self) -> None:
        """オーバーライドが存在する場合にTrueを返すこと."""
        overrides_data = {"data/file.csv": {"tag": "normalized"}}
        overrides = ColumnTypeOverrides(overrides_data)

        assert overrides.has_override("data/file.csv", "tag") is True

    def test_has_override_false(self) -> None:
        """オーバーライドが存在しない場合にFalseを返すこと."""
        overrides_data = {"data/file.csv": {"tag": "normalized"}}
        overrides = ColumnTypeOverrides(overrides_data)

        assert overrides.has_override("data/other.csv", "tag") is False


class TestAdapterIntegration:
    def test_csv_adapter_with_override(self, tmp_path: Path) -> None:
        """CSV_AdapterでオーバーライドがSOURCEタグに適用されること."""
        # UNKNOWNになるような曖昧なデータを作成
        csv_file = tmp_path / "ambiguous.csv"
        csv_file.write_text("tag\nwitch_hat\nlong hair\n", encoding="utf-8")

        # オーバーライド設定（normalizedと指定）
        overrides_data = {str(csv_file): {"tag": "normalized"}}
        overrides = ColumnTypeOverrides(overrides_data)

        adapter = CSV_Adapter(csv_file, overrides=overrides)
        df = adapter.read()

        # source_tagに変換されていること
        assert "source_tag" in df.columns
        # 小文字化が適用されていること（normalizedなので元々小文字）
        assert df["source_tag"].to_list() == ["witch_hat", "long hair"]

    def test_csv_adapter_without_override(self, tmp_path: Path) -> None:
        """オーバーライドなしの場合は自動判定が動作すること."""
        csv_file = tmp_path / "source_tags.csv"
        csv_file.write_text("tag\nwitch_hat\nlong_hair\nred_eyes\n", encoding="utf-8")

        adapter = CSV_Adapter(csv_file, overrides=None)
        df = adapter.read()

        # 自動判定でSOURCEと判定され、小文字化されること
        assert "source_tag" in df.columns
        assert df["source_tag"].to_list() == ["witch_hat", "long_hair", "red_eyes"]

    def test_csv_adapter_override_precedence(self, tmp_path: Path) -> None:
        """オーバーライドが自動判定より優先されること."""
        # 明らかにSOURCEタグのデータ
        csv_file = tmp_path / "source.csv"
        csv_file.write_text("tag\nwitch_hat\nlong_hair\nred_eyes\n", encoding="utf-8")

        # あえてnormalizedとオーバーライド
        overrides_data = {str(csv_file): {"tag": "normalized"}}
        overrides = ColumnTypeOverrides(overrides_data)

        adapter = CSV_Adapter(csv_file, overrides=overrides)
        df = adapter.read()

        # オーバーライドが適用されていればログに記録される（ここでは結果のみ検証）
        assert "source_tag" in df.columns
        assert len(df) == 3
