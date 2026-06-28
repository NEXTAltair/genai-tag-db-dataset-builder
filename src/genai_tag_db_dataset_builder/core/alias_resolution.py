"""site_tags alias conflict の解決とレポート出力.

deepghs/site_tags 由来の alias は、同じ alias が複数の canonical を指すことがある。
既定では first-win（最初に見えた canonical を採用）で解決するが、人手で確認した
ものだけ ``overrides/alias_resolution.yml`` で明示的に解決できる。

設計方針:
- conflict は build を失敗させない（収集してレポートに残すだけ）
- conflict は ``report/alias_conflicts.tsv`` に記録する
- override があれば first-win より優先する（resolution_reason=override）
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml
from loguru import logger

RESOLUTION_REASON_FIRST_WIN = "first-win"
RESOLUTION_REASON_OVERRIDE = "override"

_TSV_COLUMNS = (
    "domain",
    "alias",
    "selected_canonical",
    "rejected_canonical",
    "resolution_reason",
)


@dataclass(frozen=True)
class AliasConflict:
    """1件の alias conflict（同一 alias に複数 canonical 候補があった事象）.

    Attributes:
        domain: site_tags のサイトドメイン（例: "zerochan.net"）
        alias: 衝突した alias（source_tag）
        selected_canonical: 採用した canonical
        rejected_canonical: 不採用にした canonical
        resolution_reason: ``first-win`` または ``override``
    """

    domain: str
    alias: str
    selected_canonical: str
    rejected_canonical: str
    resolution_reason: str


class AliasResolution:
    """alias_resolution override（domain -> {alias -> canonical}）の保持クラス."""

    def __init__(self, resolution: dict[str, dict[str, str]]) -> None:
        self._resolution = resolution

    def get(self, domain: str, alias: str) -> str | None:
        """override された canonical を返す。設定が無ければ None."""
        domain_map = self._resolution.get(domain)
        if not domain_map:
            return None
        return domain_map.get(alias)

    def is_empty(self) -> bool:
        """override が一件も無いとき True."""
        return not self._resolution


def load_alias_resolution(path: Path | str | None) -> AliasResolution:
    """``overrides/alias_resolution.yml`` を読み込む.

    ファイルが無い場合は空の :class:`AliasResolution` を返す（エラーにしない）。
    トップレベルの構造が不正なら ``ValueError``、個別エントリの不備は warning で
    スキップする。

    Args:
        path: override ファイルのパス（None / 不在なら空扱い）

    Returns:
        :class:`AliasResolution`

    Raises:
        ValueError: YAML のトップ構造が mapping でない場合
    """
    if path is None:
        return AliasResolution({})

    path = Path(path)
    if not path.exists():
        logger.info(f"alias_resolution override not found (skipping): {path}")
        return AliasResolution({})

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return AliasResolution({})
    if not isinstance(data, dict):
        msg = f"alias_resolution file must be a mapping, got {type(data)}: {path}"
        raise ValueError(msg)

    raw = data.get("alias_resolution", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        msg = f"'alias_resolution' must be a mapping, got {type(raw)}: {path}"
        raise ValueError(msg)

    resolution: dict[str, dict[str, str]] = {}
    for domain, aliases in raw.items():
        if not isinstance(aliases, dict):
            logger.warning(f"alias_resolution[{domain!r}] is not a mapping, skipping")
            continue
        cleaned: dict[str, str] = {}
        for alias, canonical in aliases.items():
            if not isinstance(canonical, str) or not canonical.strip():
                logger.warning(
                    f"alias_resolution[{domain!r}][{alias!r}] has invalid canonical {canonical!r}, skipping"
                )
                continue
            cleaned[str(alias)] = canonical
        if cleaned:
            resolution[str(domain)] = cleaned

    total = sum(len(v) for v in resolution.values())
    logger.info(f"Loaded alias_resolution override: {total} entries from {path}")
    return AliasResolution(resolution)


def resolve_canonical(
    alias: str,
    candidates: Sequence[str],
    *,
    domain: str,
    resolution: AliasResolution | None,
) -> tuple[str, list[AliasConflict]]:
    """alias の最終 canonical と conflict 行を決定する.

    Args:
        alias: 対象の alias（source_tag）
        candidates: first-seen 順の一意な canonical 候補
        domain: site_tags のサイトドメイン
        resolution: override 設定（None なら first-win のみ）

    Returns:
        ``(selected_canonical, conflicts)`` のタプル。
        - 候補が1つだけで override も無い場合は conflict 無し。
        - 候補が複数の場合、override があれば override を、無ければ
          ``candidates[0]``（first-win）を採用し、不採用候補を conflict に残す。
        - 候補1つでも override が別の値を指す場合は override を採用し、
          ``resolution_reason=override`` の conflict を1件残す。
    """
    override = resolution.get(domain, alias) if resolution is not None else None

    if len(candidates) <= 1 and override is None:
        return (candidates[0] if candidates else ""), []

    if override is not None:
        selected = override
        reason = RESOLUTION_REASON_OVERRIDE
    else:
        selected = candidates[0]
        reason = RESOLUTION_REASON_FIRST_WIN

    conflicts = [
        AliasConflict(
            domain=domain,
            alias=alias,
            selected_canonical=selected,
            rejected_canonical=rejected,
            resolution_reason=reason,
        )
        for rejected in candidates
        if rejected != selected
    ]
    return selected, conflicts


def write_alias_conflicts_tsv(
    conflicts: list[AliasConflict],
    path: Path | str,
) -> Path | None:
    """alias conflict を TSV として書き出す.

    conflict が無い場合は何も書かず ``None`` を返す（CIは失敗させない）。

    Args:
        conflicts: 収集した conflict 行
        path: 出力先 TSV パス

    Returns:
        書き出したパス（conflict が無ければ None）
    """
    path = Path(path)
    if not conflicts:
        logger.info("No alias conflicts to report")
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(_TSV_COLUMNS)
        for c in conflicts:
            writer.writerow(
                [
                    c.domain,
                    c.alias,
                    c.selected_canonical,
                    c.rejected_canonical,
                    c.resolution_reason,
                ]
            )

    logger.info(f"Alias conflict report: {path} ({len(conflicts)} rows)")
    return path
