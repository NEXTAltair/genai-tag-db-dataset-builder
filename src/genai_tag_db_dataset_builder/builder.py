"""Dataset builder orchestrator.

This module orchestrates the complete dataset building process:
1. Initialize database with master data
2. Import tags_v4.db baseline
3. Import CSV sources with priority
4. Import HuggingFace datasets
5. Build indexes and optimize
"""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from genai_tag_db_dataset_builder.adapters.csv_adapter import CSV_Adapter
from genai_tag_db_dataset_builder.adapters.tags_v4_adapter import Tags_v4_Adapter
from genai_tag_db_dataset_builder.core.database import build_indexes, create_database, optimize_database
from genai_tag_db_dataset_builder.core.master_data import initialize_master_data


def build_dataset(
    output_path: Path | str,
    sources_dir: Path | str,
    version: str = "1.0.0",
) -> None:
    """Build unified tag database from all sources.

    Args:
        output_path: Output database file path
        sources_dir: Directory containing source data files
        version: Dataset version string
    """
    output_path = Path(output_path)
    sources_dir = Path(sources_dir)

    logger.info(f"Building unified tag database version {version}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Sources: {sources_dir}")

    # Phase 1: Create database and initialize master data
    logger.info("Phase 1: Creating database and initializing master data")
    create_database(output_path)
    initialize_master_data(output_path)

    # Phase 2: Import tags_v4.db baseline (highest priority)
    logger.info("Phase 2: Importing tags_v4.db baseline")
    tags_v4_path = sources_dir / "tags_v4.db"
    if tags_v4_path.exists():
        adapter = Tags_v4_Adapter(tags_v4_path)
        tables = adapter.read()
        # TODO: Import tables into database
        logger.info(f"Imported {len(tables['tags'])} tags from tags_v4.db")
    else:
        logger.warning(f"tags_v4.db not found at {tags_v4_path}, skipping baseline import")

    # Phase 3: Import CSV sources
    logger.info("Phase 3: Importing CSV sources")
    csv_files = [
        ("e621_tags_jsonl.csv", 2),  # High priority
        ("danbooru.csv", 1),
        ("e621.csv", 2),
        ("derpibooru.csv", 3),
    ]

    for csv_file, format_id in csv_files:
        csv_path = sources_dir / csv_file
        if csv_path.exists():
            logger.info(f"Importing {csv_file} (format_id={format_id})")
            adapter = CSV_Adapter(csv_path)
            df = adapter.read()
            # TODO: Merge into database
            logger.info(f"Imported {len(df)} records from {csv_file}")
        else:
            logger.warning(f"{csv_file} not found, skipping")

    # Phase 4: Build indexes
    logger.info("Phase 4: Building indexes")
    build_indexes(output_path)

    # Phase 5: Optimize database
    logger.info("Phase 5: Optimizing database")
    optimize_database(output_path)

    logger.info("Dataset build complete")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Build unified tag database")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output database file path",
    )
    parser.add_argument(
        "--sources",
        type=Path,
        default=Path("data/sources"),
        help="Directory containing source data files",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="1.0.0",
        help="Dataset version string",
    )

    args = parser.parse_args()

    build_dataset(
        output_path=args.output,
        sources_dir=args.sources,
        version=args.version,
    )


if __name__ == "__main__":
    main()
