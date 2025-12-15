"""Tags_v4_Adapter のユニット/統合テスト."""

import os
from pathlib import Path

import httpx
import pytest

from genai_tag_db_dataset_builder.adapters.tags_v4_adapter import Tags_v4_Adapter


def _default_cache_dir() -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "genai-tag-db-tools"
    return Path.home() / ".cache" / "genai-tag-db-tools"


def _resolve_tags_v4_db_path() -> Path | None:
    explicit_path = os.environ.get("GENAI_TAG_DB_TOOLS_DB_PATH")
    if explicit_path:
        path = Path(explicit_path).expanduser()
        return path if path.exists() else None

    # Monorepo workspace fallback (LoRAIro).
    lorairo_root = Path(__file__).resolve().parents[4]
    monorepo_path = (
        lorairo_root
        / "local_packages"
        / "genai-tag-db-tools"
        / "src"
        / "genai_tag_db_tools"
        / "data"
        / "tags_v4.db"
    )
    if monorepo_path.exists():
        return monorepo_path

    # Optional download (opt-in) from a direct URL (e.g. GitHub Releases asset).
    db_url = os.environ.get("GENAI_TAG_DB_TOOLS_DB_URL")
    allow_download = os.environ.get("GENAI_TAG_DB_TOOLS_ALLOW_DOWNLOAD") == "1"
    if not (db_url and allow_download):
        return None

    cache_dir = _default_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_path = cache_dir / "tags_v4.db"
    if cached_path.exists():
        return cached_path

    with httpx.stream("GET", db_url, follow_redirects=True, timeout=300) as r:
        r.raise_for_status()
        with open(cached_path, "wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)

    return cached_path


class TestTags_v4_Adapter:
    """Tags_v4_Adapterのテスト."""

    def test_init_with_nonexistent_file(self) -> None:
        """存在しないファイルでの初期化テスト."""
        with pytest.raises(FileNotFoundError):
            Tags_v4_Adapter("/nonexistent/path/tags_v4.db")

    @pytest.mark.integration
    def test_read_tables(self) -> None:
        """テーブル読み込みテスト（integration）.

        Note:
            実際のtags_v4.dbが必要なためintegrationマーカー付き
        """
        # 実際のtags_v4.dbパスを取得（相対パスで解決）
        tags_v4_path = _resolve_tags_v4_db_path()
        if tags_v4_path is None:
            pytest.skip("tags_v4.db not found")

        adapter = Tags_v4_Adapter(tags_v4_path)
        tables = adapter.read()

        # 全テーブルが存在することを確認
        assert "tags" in tables
        assert "tag_status" in tables
        assert "tag_translations" in tables
        assert "tag_usage_counts" in tables

        # TAGSテーブルの基本構造確認
        tags_df = tables["tags"]
        assert "tag_id" in tags_df.columns
        assert "tag" in tags_df.columns
        assert "source_tag" in tags_df.columns

        # レコード数確認（tags_v4.dbは993,514タグ）
        assert len(tags_df) > 900000

    @pytest.mark.integration
    def test_get_existing_tags(self) -> None:
        """既存タグ取得テスト（integration）."""
        tags_v4_path = _resolve_tags_v4_db_path()
        if tags_v4_path is None:
            pytest.skip("tags_v4.db not found")

        adapter = Tags_v4_Adapter(tags_v4_path)
        existing_tags = adapter.get_existing_tags()

        # setであることを確認
        assert isinstance(existing_tags, set)
        # レコード数確認
        assert len(existing_tags) > 900000

    @pytest.mark.integration
    def test_get_next_tag_id(self) -> None:
        """次のtag_id取得テスト（integration）."""
        tags_v4_path = _resolve_tags_v4_db_path()
        if tags_v4_path is None:
            pytest.skip("tags_v4.db not found")

        adapter = Tags_v4_Adapter(tags_v4_path)
        next_id = adapter.get_next_tag_id()

        # 正の整数であることを確認
        assert isinstance(next_id, int)
        assert next_id > 0

    def test_deduplicate_tags_removes_duplicates(self) -> None:
        """重複タグが除去されることを確認."""
        import polars as pl

        # 重複タグを含むモックDataFrame（tag_id=1, 5でtagが同じ）
        tags_df = pl.DataFrame(
            {
                "tag_id": [1, 2, 3, 5, 10],
                "tag": ["witch", "mage", "spiked collar", "witch", "wizard"],
                "source_tag": ["Witch", "Mage", "Spiked_Collar", "witch", "Wizard"],
            }
        )

        adapter = Tags_v4_Adapter(__file__)  # Pathは使用されないのでダミー
        deduplicated = adapter._deduplicate_tags(tags_df)

        # "witch"の重複が除去され、最小tag_id (1) のみが残る
        assert len(deduplicated) == 4
        witch_rows = deduplicated.filter(pl.col("tag") == "witch")
        assert len(witch_rows) == 1
        assert witch_rows["tag_id"][0] == 1
        assert witch_rows["source_tag"][0] == "witch"
