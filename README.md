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