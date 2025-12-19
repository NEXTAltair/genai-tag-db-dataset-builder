"""CSV読み込みアダプタ（修復付き）.

壊れたCSV（余計なカンマ、欠損列など）も可能な範囲で読み込み・修復します。
"""

from pathlib import Path

import polars as pl
from loguru import logger

from genai_tag_db_dataset_builder.core.column_classifier import TagColumnType, classify_tag_column
from genai_tag_db_dataset_builder.core.exceptions import NormalizedSourceSkipError
from genai_tag_db_dataset_builder.core.normalize import canonicalize_source_tag
from genai_tag_db_dataset_builder.core.overrides import ColumnTypeOverrides

from .base_adapter import BaseAdapter


class CSV_Adapter(BaseAdapter):
    """汎用CSVアダプタ（壊れたCSVの修復機能付き）.

    Args:
        file_path: CSVファイルのパス
        repair_mode: 修復モード（"derpibooru", "dataset_rising_v2", "english_dict"）
    """

    def __init__(
        self,
        file_path: Path | str,
        repair_mode: str | None = None,
        unknown_report_dir: Path | str | None = None,
        overrides: ColumnTypeOverrides | None = None,
    ) -> None:
        """アダプタ初期化.

        Args:
            file_path: CSVファイルのパス
            repair_mode: 特定の壊れたCSV向けの修復モード
            unknown_report_dir: tag列の意味推定が UNKNOWN の場合に出力するレポート先ディレクトリ
            overrides: 列タイプ手動オーバーライド設定（Noneの場合は自動判定のみ）

        Raises:
            FileNotFoundError: CSVファイルが存在しない場合
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.file_path}")

        self.repair_mode = repair_mode
        self.unknown_report_dir = Path(unknown_report_dir) if unknown_report_dir else None
        self.overrides = overrides

    def read(self) -> pl.DataFrame:
        """CSVファイルを読み込む.

        Returns:
            正規化された Polars DataFrame

        Raises:
            ValueError: CSV読み込みに失敗した場合
        """
        try:
            df = pl.read_csv(
                self.file_path,
                ignore_errors=True,
                truncate_ragged_lines=True,
            )
        except Exception as e:
            raise ValueError(f"Failed to read CSV: {self.file_path}") from e

        if self.repair_mode:
            df = self.repair(df)

        df = self._normalize_columns(df)
        return df

    def _normalize_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """列名とsource_tag表記を標準形に正規化する.

        - 入力が `tag` 列しか持たない場合、`tag` 列の意味（SOURCE/NORMALIZED/UNKNOWN）を推定しつつ
          互換性のため `source_tag` に寄せる
        - overrides設定がある場合は自動判定より優先される
        - `source_tag` は DB保存向けに小文字統一する（顔文字はそのまま）
        """
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
                # dataset_rising_v2 は「正則化済み tag + deprecated_tags」の辞書で、source_tag が無い。
                # この場合に限り、tag を source_tag として扱って取り込みを継続する。
                if (
                    decision == TagColumnType.NORMALIZED
                    and self.repair_mode == "dataset_rising_v2"
                    and "deprecated_tags" in df.columns
                ):
                    logger.warning(
                        f"{self.file_path}: tag列はNORMALIZED推定だが、dataset_rising_v2として取り込みを継続します。"
                        "（tag列を source_tag として扱い、deprecated_tags で alias 情報を取り込む）"
                    )
                else:
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
        """UNKNOWN判定のレポートをTSVで出力する（人力修正用）."""
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
        """壊れたデータを修復する（repair_modeが指定されている場合）."""
        if self.repair_mode == "derpibooru":
            return self._repair_derpibooru(df)
        if self.repair_mode == "dataset_rising_v2":
            return self._repair_dataset_rising_v2(df)
        if self.repair_mode == "english_dict":
            return self._repair_english_dict(df)
        return df

    def _repair_derpibooru(self, df: pl.DataFrame) -> pl.DataFrame:
        """derpibooru.csv向け修復: format_id欠損を補う."""
        if "format_id" not in df.columns:
            df = df.with_columns(pl.lit(3).alias("format_id"))
        return df

    def _repair_dataset_rising_v2(self, df: pl.DataFrame) -> pl.DataFrame:
        """dataset_rising_v2.csv向け修復: 余計なカンマで生成された空列を削除する."""
        cols_to_keep: list[str] = []
        for col in df.columns:
            null_ratio = df[col].is_null().sum() / len(df)
            if null_ratio < 0.9:
                cols_to_keep.append(col)

        return df.select(cols_to_keep)

    def _repair_english_dict(self, df: pl.DataFrame) -> pl.DataFrame:
        """EnglishDictionary.csv向け修復: fomat_id → format_id にリネームする."""
        if "fomat_id" in df.columns:
            df = df.rename({"fomat_id": "format_id"})
        return df
