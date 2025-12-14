"""JSON_Adapter for reading JSON files.

This adapter handles JSON file reading for tag data sources.
"""

import json
from pathlib import Path

import polars as pl

from .base_adapter import BaseAdapter


class JSON_Adapter(BaseAdapter):
    """Adapter for JSON tag sources.

    Args:
        file_path: Path to JSON file
    """

    def __init__(self, file_path: Path | str) -> None:
        """Initialize adapter.

        Args:
            file_path: Path to JSON file

        Raises:
            FileNotFoundError: JSON file does not exist
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"JSON file not found: {self.file_path}")

    def read(self) -> pl.DataFrame:
        """Read a JSON file into a Polars DataFrame.

        Returns:
            Parsed Polars DataFrame

        Raises:
            ValueError: Failed to read JSON
        """
        try:
            with open(self.file_path, encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                data = [data]

            return pl.DataFrame(data)

        except Exception as e:
            raise ValueError(f"Failed to read JSON: {self.file_path}") from e

    def validate(self, df: pl.DataFrame) -> bool:
        """Validate a DataFrame."""
        if df.is_empty():
            return False
        return True

    def repair(self, df: pl.DataFrame) -> pl.DataFrame:
        """Repair data if needed (no-op for JSON)."""
        return df
