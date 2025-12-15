"""Dataset builder exceptions.

カスタム例外クラスを定義します。
"""


class NormalizedSourceSkipError(Exception):
    """非SOURCE（NORMALIZED/UNKNOWN）と判定されたソースをスキップする例外.

    基本データソースは正則化していない前提のため、SOURCEと断定できない（NORMALIZED/UNKNOWN）
    ソースは誤統合を防ぐために全体スキップします。

    Attributes:
        file_path: スキップ対象のファイルパス
        decision: 判定結果（TagColumnType.NORMALIZED または TagColumnType.UNKNOWN）
        signals: 判定シグナル詳細
    """

    def __init__(self, file_path: str, decision: str, signals: dict) -> None:
        """例外初期化.

        Args:
            file_path: スキップ対象のファイルパス
            decision: 判定結果
            signals: 判定シグナル詳細
        """
        self.file_path = file_path
        self.decision = decision
        self.signals = signals
        message = (
            f"Source skipped: {file_path} (decision={decision}). "
            "Non-SOURCE tags detected (NORMALIZED/UNKNOWN), which may cause incorrect merging. "
            f"Signals: {signals}. "
            "To force import, add override in column_type_overrides.json"
        )
        super().__init__(message)
