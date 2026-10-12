"""Shared translation-value normalization for source import and enrichment."""

import re

_SEPARATORS = re.compile("[,\uff0c\u3001\uff64\ufe50]")


def split_translation_text(text: str) -> list[str]:
    parts = [
        part.strip().strip(" \"'\u201c\u201d\u2018\u2019\u300c\u300d") for part in _SEPARATORS.split(text)
    ]
    return [part for part in parts if part]
