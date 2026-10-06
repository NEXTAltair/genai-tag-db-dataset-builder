from __future__ import annotations

import pytest

from genai_tag_db_dataset_builder.core.scripts import guess_language_by_script


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("オリジナル", "ja"),
        ("うちの子", "ja"),
        ("ラーメン", "ja"),
        ("창작", "ko"),
        ("原創", "zh"),
        # 記号だけの U+30FB / U+30FC / U+30A0 は仮名とみなさない
        ("原創・", "zh"),
        ("原創ー", "zh"),
        ("原創\u30a0", "zh"),
        ("oc", None),
        ("Оригинальный", None),
    ],
)
def test_guess_language_by_script(text: str, expected: str | None) -> None:
    assert guess_language_by_script(text) == expected
