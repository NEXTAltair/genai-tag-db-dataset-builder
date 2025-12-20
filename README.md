# genai-tag-db-dataset-builder

Dataset builder for unified generative AI tag database.
生成AI向け統合タグデータベースを構築するためのビルダーです。

## Overview

This package builds a unified tag database by merging multiple data sources (tags_v4.db, HuggingFace datasets, local CSV files) into a single optimized SQLite database compatible with genai-tag-db-tools.
複数のデータソース（tags_v4.db / Hugging Face datasets / ローカルCSV）を統合し、genai-tag-db-tools と互換なSQLiteデータベースを生成します。

## Features

- **Multi-source integration**: Merges tags from Danbooru, E621, Derpibooru, and other platforms / 複数サイトのタグを統合
- **Data repair**: Automatically fixes broken CSV files and data inconsistencies / CSVの破損や不整合を自動修復
- **Conflict detection**: Generates CSV reports for manual review of type_id conflicts and alias changes / 競合レポートをCSV出力
- **SQLite optimization**: Applies build-time and distribution-time PRAGMA settings for performance / SQLite最適化PRAGMAを適用
- **Schema compatibility**: 100% compatible with genai-tag-db-tools database schema / genai-tag-db-tools互換スキーマ

## Installation

```bash
# From LoRAIro project root
cd /workspaces/LoRAIro
uv sync
```

## Usage

```python
from genai_tag_db_dataset_builder.core.normalize import normalize_tag
from genai_tag_db_dataset_builder.adapters import BaseAdapter

# Normalize tags
normalized = normalize_tag("spiked_collar")  # Returns: "spiked collar"

# Build dataset (implementation in progress)
# See design plan: .serena/memories/dataset_builder_design_plan_2025_12_13.md
```

## Development

```bash
# Run tests
uv run pytest local_packages/genai-tag-db-dataset-builder/tests/

# Run specific test category
uv run pytest local_packages/genai-tag-db-dataset-builder/tests/unit/ -m unit

# Format code
uv run ruff format src/ tests/
uv run ruff check src/ tests/ --fix

# Type checking
uv run mypy src/
```

## Architecture

```
src/genai_tag_db_dataset_builder/
├── adapters/           # Data source adapters (CSV, JSON, Parquet, SQLite)
├── core/               # Core functionality (normalize, merge, conflict detection)
└── utils/              # Utility functions

tests/
├── unit/               # Unit tests (fast, isolated)
└── integration/        # Integration tests (requires external resources)
```

## Design Documentation

- **Main Design Plan**: `.serena/memories/dataset_builder_design_plan_2025_12_13.md`
- **Core Algorithm Fix**: `.serena/memories/dataset_builder_core_algorithm_fix_2025_12_13.md`
設計資料は `.serena/memories/` 以下に集約されています。

## Implementation Status

**Phase 0: Foundation Setup** (Week 1)
- [x] Package name unification: genai-tag-db-dataset-builder
- [x] pyproject.toml configuration
- [x] Basic directory structure
- [x] README.md documentation

**Phase 1: Adapter Implementation** 🚧 (Week 2-3)
- [ ] BaseAdapter abstract class
- [ ] Tags_v4_Adapter implementation
- [ ] CSV_Adapter implementation (with repair logic)
- [ ] JSON_Adapter implementation
- [ ] Parquet_Adapter implementation

**Phase 2: Merge Logic Implementation** (Week 4-5)
- [ ] merge_tags() implementation (set difference approach)
- [ ] process_deprecated_tags() implementation (alias generation)
- [ ] detect_conflicts() implementation (tag + format_id JOIN)

See design plan for full implementation roadmap.
詳細なロードマップは設計資料を参照してください。

## License

MIT / MITライセンス
