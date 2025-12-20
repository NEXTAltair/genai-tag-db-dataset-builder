"""列タイプ手動オーバーライド機能.

UNKNOWN判定された列に対して、ビルド時に手動で列タイプを指定できる機能を提供します。

使用例:
    >>> overrides = load_overrides(Path("column_type_overrides.json"))
    >>> col_type = get_override(overrides, "data/tags.csv", "tag")
    >>> if col_type:
    ...     print(f"Override: {col_type}")
"""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

from .column_classifier import TagColumnType


class ColumnTypeOverrides:
    """列タイプオーバーライド管理クラス.

    JSON形式のオーバーライド設定を読み込み、ファイル+列名の組み合わせで
    列タイプを手動指定できる機能を提供します。

    JSON形式:
        {
            "path/to/file.csv": {
                "tag": "NORMALIZED"
            },
            "path/to/other.json": {
                "tag": "SOURCE"
            }
        }
    """

    def __init__(self, overrides: dict[str, dict[str, str]]) -> None:
        """オーバーライド設定を初期化.

        Args:
            overrides: ファイルパス -> {列名 -> 列タイプ} のマッピング

        Raises:
            ValueError: 無効な列タイプが含まれている場合
        """
        self._overrides = overrides
        self._validate()

    def _validate(self) -> None:
        """オーバーライド設定の妥当性を検証.

        Raises:
            ValueError: 無効な列タイプが含まれている場合
        """
        valid_types = {t.value for t in TagColumnType}

        for file_path, columns in self._overrides.items():
            if not isinstance(columns, dict):
                msg = f"Invalid override format for '{file_path}': expected dict, got {type(columns)}"
                raise ValueError(msg)

            for column_name, column_type in columns.items():
                if column_type not in valid_types:
                    msg = (
                        f"Invalid column type '{column_type}' for '{file_path}:{column_name}'. "
                        f"Valid types: {valid_types}"
                    )
                    raise ValueError(msg)

    def get(self, file_path: str | Path, column_name: str) -> TagColumnType | None:
        """指定されたファイル+列名のオーバーライドを取得.

        Args:
            file_path: ファイルパス（相対または絶対）
            column_name: 列名

        Returns:
            オーバーライドされた列タイプ、設定がない場合は None
        """
        file_path_str = str(file_path)

        # 完全一致を試す
        if file_path_str in self._overrides:
            columns = self._overrides[file_path_str]
            if column_name in columns:
                return TagColumnType(columns[column_name])

        # パス正規化して再試行（相対パス対応）
        normalized_path = Path(file_path_str).as_posix()
        for override_path, columns in self._overrides.items():
            if Path(override_path).as_posix() == normalized_path and column_name in columns:
                return TagColumnType(columns[column_name])

        return None

    def has_override(self, file_path: str | Path, column_name: str) -> bool:
        """指定されたファイル+列名のオーバーライドが存在するか確認.

        Args:
            file_path: ファイルパス
            column_name: 列名

        Returns:
            オーバーライドが存在する場合 True
        """
        return self.get(file_path, column_name) is not None


def load_overrides(overrides_path: Path | str) -> ColumnTypeOverrides:
    """JSONファイルからオーバーライド設定を読み込む.

    Args:
        overrides_path: オーバーライド設定JSONファイルのパス

    Returns:
        オーバーライド設定オブジェクト

    Raises:
        FileNotFoundError: ファイルが存在しない場合
        ValueError: JSON形式が不正、または無効な列タイプが含まれている場合
    """
    overrides_path = Path(overrides_path)

    if not overrides_path.exists():
        raise FileNotFoundError(f"Overrides file not found: {overrides_path}")

    try:
        with open(overrides_path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        msg = f"Invalid JSON in overrides file: {overrides_path}"
        raise ValueError(msg) from e

    if not isinstance(data, dict):
        msg = f"Overrides file must contain a JSON object, got {type(data)}"
        raise ValueError(msg)

    logger.info(f"Loaded {len(data)} file overrides from {overrides_path}")
    return ColumnTypeOverrides(data)


def get_override(
    overrides: ColumnTypeOverrides | None,
    file_path: str | Path,
    column_name: str,
) -> TagColumnType | None:
    """オーバーライド設定から列タイプを取得（ヘルパー関数）.

    Args:
        overrides: オーバーライド設定オブジェクト（Noneの場合は常にNoneを返す）
        file_path: ファイルパス
        column_name: 列名

    Returns:
        オーバーライドされた列タイプ、設定がない場合は None
    """
    if overrides is None:
        return None
    return overrides.get(file_path, column_name)
