"""Parquet読み込みアダプタ.

HuggingFace datasets（例: deepghs/site_tags、isek-ai/danbooru-wiki-2024）などの
Parquetファイルを読み込みます。
"""

from pathlib import Path

import polars as pl
from loguru import logger

from genai_tag_db_dataset_builder.core.column_classifier import TagColumnType, classify_tag_column
from genai_tag_db_dataset_builder.core.exceptions import NormalizedSourceSkipError
from genai_tag_db_dataset_builder.core.normalize import canonicalize_source_tag
from genai_tag_db_dataset_builder.core.overrides import ColumnTypeOverrides

from .base_adapter import BaseAdapter


class Parquet_Adapter(BaseAdapter):
    """Parquetファイル用アダプタ."""

    def __init__(
        self,
        file_path: Path | str,
        unknown_report_dir: Path | str | None = None,
        overrides: ColumnTypeOverrides | None = None,
    ) -> None:
        """アダプタ初期化.

        Args:
            file_path: Parquetファイルのパス
            unknown_report_dir: tag列の意味推定が UNKNOWN の場合に出力するレポート先ディレクトリ
            overrides: 列タイプ手動オーバーライド設定（Noneの場合は自動判定のみ）

        Raises:
            FileNotFoundError: Parquetファイルが存在しない場合
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {self.file_path}")

        self.unknown_report_dir = Path(unknown_report_dir) if unknown_report_dir else None
        self.overrides = overrides

    def _normalize_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """列名とsource_tag表記を標準形に正規化する（tag → source_tag 等）."""
        if "tag" in df.columns and "source_tag" not in df.columns:
            # overrides優先チェック
            override_type = self.overrides.get(self.file_path, "tag") if self.overrides else None

            if override_type:
                decision = override_type
                signals = {"override": True, "decision": override_type.value}
                logger.info(f"Applied override for {self.file_path}:tag -> {override_type.value}")
            else:
                try:
                    decision, signals = classify_tag_column(df, "tag")
                except Exception as e:
                    decision, signals = TagColumnType.UNKNOWN, {"error": str(e)}

            # NORMALIZED/UNKNOWN判定時のスキップ処理（overridesがない場合のみ）
            if decision in (TagColumnType.NORMALIZED, TagColumnType.UNKNOWN) and not override_type:
                # UNKNOWN の場合はレポート出力
                if decision == TagColumnType.UNKNOWN and self.unknown_report_dir:
                    self._export_unknown_column_report(
                        source_path=self.file_path,
                        column_name="tag",
                        signals=signals,
                    )

                logger.error(
                    f"取り込みスキップ: {self.file_path} (decision={decision.value}). "
                    "取り込み対象は SOURCE のみです。"
                    "なお、丸括弧エスケープ（`\\(` `\\)`）は正則化後に付与されるため NORMALIZED の強いシグナルです。"
                    f" 根拠: {signals}"
                )
                raise NormalizedSourceSkipError(
                    file_path=str(self.file_path),
                    decision=decision.value,
                    signals=signals,
                )

            df = df.rename({"tag": "source_tag"})
            logger.debug(
                f"Normalized 'tag' column to 'source_tag' (decision={decision}, signals={signals})"
            )

        if "source_tag" in df.columns:
            df = df.with_columns(
                pl.col("source_tag")
                .map_elements(canonicalize_source_tag, return_dtype=pl.String)
                .alias("source_tag")
            )

        return df


    def _export_unknown_column_report(
        self,
        source_path: Path,
        column_name: str,
        signals: dict[str, object],
    ) -> None:
        if not self.unknown_report_dir:
            return

        self.unknown_report_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.unknown_report_dir / f"{source_path.stem}__unknown_{column_name}.tsv"

        def fmt(v: object) -> str:
            s = "" if v is None else str(v)
            return s.replace("\t", " ").replace("\r", " ").replace("\n", " ")

        header = [
            "source_file",
            "column_name",
            "decision",
            "confidence",
            "underscore_ratio",
            "escaped_paren_ratio",
            "normalize_change_ratio",
            "notes",
        ]

        row = [
            str(source_path),
            column_name,
            fmt(signals.get("decision", "unknown")),
            fmt(signals.get("confidence", "")),
            fmt(signals.get("underscore_ratio", "")),
            fmt(signals.get("escaped_paren_ratio", "")),
            fmt(signals.get("normalize_change_ratio", "")),
            fmt(signals.get("error", "")),
        ]

        out_path.write_text("\t".join(header) + "\n" + "\t".join(row) + "\n", encoding="utf-8")

    def read(self) -> pl.DataFrame:
        """Parquetファイルを読み込む.

        Returns:
            読み込んだ DataFrame

        Raises:
            ValueError: Parquet読み込みに失敗した場合
        """
        try:
            df = pl.read_parquet(self.file_path)
        except Exception as e:
            raise ValueError(f"Failed to read Parquet: {self.file_path}") from e

        df = self._normalize_columns(df)
        return df

    def validate(self, df: pl.DataFrame) -> bool:
        """データ整合性検証."""
        if df.is_empty():
            return False

        # source_tag列の存在確認（正規化後は必須）
        if "source_tag" not in df.columns:
            logger.error("Missing required 'source_tag' column after normalization")
            return False

        return True

    def repair(self, df: pl.DataFrame) -> pl.DataFrame:
        """データ修復（Parquetは基本的に不要なのでそのまま返す）."""
        return df
