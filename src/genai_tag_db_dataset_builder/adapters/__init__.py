"""タグDB構築用のデータソースアダプタ群."""

from .base_adapter import STANDARD_COLUMNS, BaseAdapter
from .csv_adapter import CSV_Adapter
from .json_adapter import JSON_Adapter
from .parquet_adapter import Parquet_Adapter
from .tags_v4_adapter import Tags_v4_Adapter

__all__ = [
    "BaseAdapter",
    "CSV_Adapter",
    "JSON_Adapter",
    "Parquet_Adapter",
    "Tags_v4_Adapter",
    "STANDARD_COLUMNS",
]
