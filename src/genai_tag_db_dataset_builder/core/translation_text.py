"""Shared translation-value normalization for source import and enrichment."""

import re

_SEPARATORS = re.compile("[,\uff0c\uff64\ufe50]")
_SEPARATORS_WITH_IDEOGRAPHIC = re.compile("[,\uff0c\u3001\uff64\ufe50]")


def split_translation_text(text: str, *, ideographic_comma: bool = False) -> list[str]:
    """カンマで訳を分割する。読点は題名に含まれるので、列挙と分かる入力でだけ区切りにする."""
    separators = _SEPARATORS_WITH_IDEOGRAPHIC if ideographic_comma else _SEPARATORS
    parts = [
        part.strip().strip(" \"'\u201c\u201d\u2018\u2019\u300c\u300d") for part in separators.split(text)
    ]
    return [part for part in parts if part]
