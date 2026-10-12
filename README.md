# genai-tag-db-dataset-builder

Dataset builder for a unified generative-AI tag database.
生成AI向け統合タグデータベースを構築するためのビルダーです。

## Overview / 概要

Builds a unified SQLite tag database from multiple sources (SQLite/CSV/JSON/Parquet/HF datasets).
複数ソース（SQLite/CSV/JSON/Parquet/HF datasets）から統合SQLiteを生成します。

## Features / 特徴

- Multi-source integration / 複数ソース統合
- Data repair and validation / 修復・検証
- Conflict reporting / 競合レポート出力
- Build-time SQLite optimizations / SQLite最適化

## Installation / インストール

```bash
# Clone and enter the repo
git clone https://github.com/NEXTAltair/genai-tag-db-dataset-builder.git
cd genai-tag-db-dataset-builder

# Install dependencies
uv sync
```

## CLI Usage / 実行例

```bash
# Basic build (sources are under the given directory)
uv run python -m genai_tag_db_dataset_builder.builder \
  --output ./out_db/genai_tag_db.sqlite \
  --sources ./data_sources \
  --report-dir ./out_db \
  --overwrite
```

```bash
# Build by extending a base database
uv run python -m genai_tag_db_dataset_builder.builder \
  --output ./out_db/genai_tag_db.sqlite \
  --sources ./data_sources \
  --report-dir ./out_db \
  --base-db ./base/genai-image-tag-db-cc0.sqlite \
  --overwrite
```

## Legacy tag-sheet enrichment / 旧タグ表の取り込み

CC0 の定期ビルドは `data/tag_table_bundle.json` を取り込みます。ファイル内容の SHA-256 を
ソース revision として記録し、更新時には再ビルドします。MIT/CC4 には CC0 の追加分が継承されます。

ローカルの CC0 コピーへ反映する場合:

```bash
python -m genai_tag_db_dataset_builder.builder \
  --base-db ./base/genai-image-tag-db-cc0.sqlite \
  --output ./out_db/genai-image-tag-db-cc0.sqlite \
  --sources ./empty_sources --no-site-tags \
  --tag-table-bundle data/tag_table_bundle.json \
  --report-dir ./out_db/report --version v2026.10.12.tag-table
```

`empty_sources` は事前に作成してください。入力DBと出力DBには別のパスを指定します。
日本語訳は既存の訳がないタグだけ補完し、訳のある alias 先を持つタグも補完対象から除きます。
新規タグは unknown format/type で登録し、種別や使用回数は推定しません。
取り込みは追加のみで、同じバンドルの再実行は変更ゼロです。不正な階層・参照は反映前に拒否し、
既存分類との衝突はタグ追加を含めてロールバックします。

分類は booru のタグ種別と独立した以下のテーブルで保持します:

- `TAG_CATEGORIES`: namespace、親、ラベル、階層の JSON path。
- `TAG_CATEGORY_MEMBERSHIPS`: タグと分類の多対多対応。
- `TAG_SOURCE_REFERENCES` / `TAG_CATEGORY_SOURCES`: バンドルID、元TSVの相対パス、行番号。

分類配下のタグを検索する SQL（`:category_id` を対象分類IDに設定）:

```sql
WITH RECURSIVE subtree(category_id) AS (
  SELECT category_id FROM TAG_CATEGORIES WHERE category_id = :category_id
  UNION ALL
  SELECT c.category_id FROM TAG_CATEGORIES c JOIN subtree s ON c.parent_id = s.category_id
)
SELECT DISTINCT t.tag
FROM subtree s JOIN TAG_CATEGORY_MEMBERSHIPS m USING (category_id)
JOIN TAGS t USING (tag_id)
ORDER BY t.tag;
```

結果は SQLite に入ります。現在の Danbooru 用 Parquet と GUI の分類検索には追加しません。
取り込み件数・既存訳の保持・複数候補・文字種による除外は `report/tag_table_<bundle_id>.json` に出力します。
バンドルの再生成方法と対象表は [data/README.md](data/README.md) を参照してください。

## Development / 開発

```bash
# Unit tests
uv run pytest tests/unit/ -v -m "not integration"

# Lint & format
uv run ruff format .
uv run ruff check .

# Type checking
uv run mypy src/genai_tag_db_dataset_builder
```

## Scheduled Dataset Refresh / 定期データセット更新

`.github/workflows/update-tag-dataset-ci.yml` rebuilds and publishes the tag databases weekly:

- Schedule: Monday 03:00 JST (Sunday 18:00 UTC)
- Target on schedule: `all` (CC0, MIT, and CC4)
- Publish destination (Hugging Face Datasets):
  - [NEXTAltair/genai-image-tag-db](https://huggingface.co/datasets/NEXTAltair/genai-image-tag-db)
  - [NEXTAltair/genai-image-tag-db-mit](https://huggingface.co/datasets/NEXTAltair/genai-image-tag-db-mit)
  - [NEXTAltair/genai-image-tag-db-CC4](https://huggingface.co/datasets/NEXTAltair/genai-image-tag-db-CC4)
- Required secret: `HF_TOKEN`

Manual runs can still choose a narrower target (`cc0`, `mit`, `cc4`, or `all`) and `force`.

## Layout / 構成

```
src/genai_tag_db_dataset_builder/
├── adapters/           # Data source adapters (CSV/JSON/Parquet/SQLite/HF)
├── core/               # Normalize/merge/conflict detection
└── tools/              # Build helpers and reports

tests/
├── unit/
└── integration/
```

## Notes / 補足

Project design notes and work logs are maintained outside this README.
設計メモや作業ログはREADMEとは別で管理します。

## License

MIT
