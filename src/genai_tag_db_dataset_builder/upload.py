"""HuggingFace dataset uploader.

This module handles uploading the built database and metadata to HuggingFace Hub.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, create_repo
from loguru import logger


def upload_to_huggingface(
    db_path: Path | str,
    metadata_path: Path | str,
    repo_id: str,
    version: str,
    token: str,
    private: bool = False,
) -> str:
    """Upload dataset to HuggingFace Hub.

    Args:
        db_path: Path to the built database
        metadata_path: Path to metadata JSON
        repo_id: HuggingFace repository ID (e.g., 'username/repo-name')
        version: Dataset version string
        token: HuggingFace API token
        private: Whether to create a private repository

    Returns:
        URL to the uploaded dataset
    """
    db_path = Path(db_path)
    metadata_path = Path(metadata_path)

    logger.info(f"Uploading dataset to HuggingFace: {repo_id}")
    logger.info(f"Version: {version}")

    # Initialize HF API
    api = HfApi(token=token)

    # Create repository if it doesn't exist
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True,
            token=token,
        )
        logger.info(f"Repository {repo_id} ready")
    except Exception as e:
        logger.error(f"Failed to create repository: {e}")
        raise

    # Upload database file
    logger.info(f"Uploading database: {db_path.name}")
    db_url = api.upload_file(
        path_or_fileobj=str(db_path),
        path_in_repo=f"{version}/unified_tags.db",
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
    )
    logger.info(f"Database uploaded: {db_url}")

    # Upload metadata
    logger.info(f"Uploading metadata: {metadata_path.name}")
    metadata_url = api.upload_file(
        path_or_fileobj=str(metadata_path),
        path_in_repo=f"{version}/metadata.json",
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
    )
    logger.info(f"Metadata uploaded: {metadata_url}")

    # Generate and upload README
    readme_content = generate_readme(metadata_path, version)
    readme_path = db_path.parent / "README.md"
    readme_path.write_text(readme_content, encoding="utf-8")

    logger.info("Uploading README")
    readme_url = api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
    )
    logger.info(f"README uploaded: {readme_url}")

    dataset_url = f"https://huggingface.co/datasets/{repo_id}"
    logger.info(f"Upload complete: {dataset_url}")

    return dataset_url


def generate_readme(metadata_path: Path, version: str) -> str:
    """Generate README content from metadata.

    Args:
        metadata_path: Path to metadata JSON
        version: Dataset version

    Returns:
        README markdown content
    """
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    stats = metadata["statistics"]

    readme = f"""---
license: mit
task_categories:
  - text-generation
  - image-classification
language:
  - en
  - ja
tags:
  - tags
  - anime
  - booru
  - danbooru
  - e621
  - derpibooru
size_categories:
  - 1M<n<10M
---

# Unified Tag Database

A comprehensive, unified tag database compiled from multiple sources including Danbooru, E621, Derpibooru, and other booru-style websites.

## Version

**{version}** - Built on {metadata['build_date']}

## Statistics

- **Total Tags**: {stats['total_tags']:,}
- **Total Formats**: {stats['total_formats']}
- **Total Type Names**: {stats['total_type_names']}
- **Total Translations**: {stats['total_translations']:,}
- **Database Size**: {stats['database_size_mb']} MB

### Per-Format Statistics

| Format | Tags |
|--------|------|
"""

    for fmt in stats["format_statistics"]:
        readme += f"| {fmt['format_name']} | {fmt['tag_count']:,} |\n"

    readme += f"""

## Schema

This database follows the genai-tag-db-tools schema with the following tables:

### Core Tables

- **TAGS**: Unified tag definitions with normalized tag names
- **TAG_STATUS**: Tag metadata per format (type, alias, preferred tag)
- **TAG_TRANSLATIONS**: Multi-language tag translations
- **TAG_USAGE_COUNTS**: Tag usage statistics per format

### Master Data Tables

- **TAG_FORMATS**: Format definitions (Danbooru, E621, Derpibooru, etc.)
- **TAG_TYPE_NAME**: Type name definitions (general, artist, character, etc.)
- **TAG_TYPE_FORMAT_MAPPING**: Type ID to type name mapping per format

## Usage

### Python (SQLite)

```python
import sqlite3

conn = sqlite3.connect("unified_tags.db")

# Search for tags
cursor = conn.execute("SELECT * FROM TAGS WHERE tag LIKE ?", ("%witch%",))
for row in cursor:
    print(row)

conn.close()
```

### Python (Polars)

```python
import polars as pl

# Read tags table
tags_df = pl.read_database("SELECT * FROM TAGS", "sqlite:///unified_tags.db")
print(tags_df)
```

### Python (genai-tag-db-tools)

```python
from genai_tag_db_tools.data.tag_repository import TagRepository

repo = TagRepository("unified_tags.db")

# Search tags
results = repo.search_tags_by_name("witch")
for tag in results:
    print(tag.tag, tag.source_tag)
```

## Data Sources

This dataset is compiled from:

- **tags_v4.db**: Baseline data from genai-tag-db-tools
- **deepghs/site_tags**: Latest tag data from multiple booru sites
- **isek-ai/danbooru-wiki-2024**: Translation and wiki data
- **TagDB_DataSource_CSV**: Various CSV exports from booru sites

## License

MIT License

## Citation

If you use this dataset in your research or project, please cite:

```bibtex
@misc{{unified-tag-database-{version.replace('.', '-')},
  author = {{NEXTAltair}},
  title = {{Unified Tag Database}},
  year = {{{metadata['build_date'][:4]}}},
  version = {{{version}}},
  url = {{https://huggingface.co/datasets/NEXTAltair/unified-tag-database}}
}}
```

## Acknowledgments

Special thanks to:
- deepghs for site_tags dataset
- isek-ai for danbooru-wiki dataset
- All contributors to booru-style tagging systems
"""

    return readme


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Upload dataset to HuggingFace")
    parser.add_argument(
        "--db",
        type=Path,
        required=True,
        help="Path to the built database",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="Path to metadata JSON",
    )
    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="HuggingFace repository ID (e.g., 'username/repo-name')",
    )
    parser.add_argument(
        "--version",
        type=str,
        required=True,
        help="Dataset version string",
    )
    parser.add_argument(
        "--token",
        type=str,
        required=True,
        help="HuggingFace API token",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create private repository",
    )

    args = parser.parse_args()

    upload_to_huggingface(
        db_path=args.db,
        metadata_path=args.metadata,
        repo_id=args.repo,
        version=args.version,
        token=args.token,
        private=args.private,
    )


if __name__ == "__main__":
    main()
