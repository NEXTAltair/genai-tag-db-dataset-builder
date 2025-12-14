"""Tag normalization functions.

This module provides tag normalization functionality for converting source tags
from various formats into the standardized TAGS.tag format.
"""

from __future__ import annotations

import re

_UNESCAPED_LPAREN = re.compile(r"(?<!\\)\(")
_UNESCAPED_RPAREN = re.compile(r"(?<!\\)\)")
_SIMPLE_EMOTICON = re.compile(r"^\s*[:;=8xX]-?[)D(P/\\|]\s*$", re.IGNORECASE)
_COLON_ANGLE = re.compile(r"^\s*[:;][<>]=?\s*$")  # :<, :>, :>=, ;<, ;>
_DISAMBIGUATION = re.compile(r"^[a-z0-9][a-z0-9_./:-]*_\([a-z0-9][a-z0-9_./:-]*\)$", re.IGNORECASE)
_SLASH_TAG = re.compile(r"^/[a-z0-9][a-z0-9_./:-]*/$", re.IGNORECASE)
_PARENS_WORD = re.compile(r"^\([a-z0-9_]{2,}\)$", re.IGNORECASE)

# Characters commonly used in short ASCII kaomoji/emoticons.
_FACE_CHARS = r"0-9oOxXtTuUvV\.\-\+;:\^<>=_"
_FACE_UNDERSCORE = re.compile(rf"^[{_FACE_CHARS}]{{1,3}}(?:_[{_FACE_CHARS}]{{1,3}})+$")
_FACE_PARENS = re.compile(rf"^\([{_FACE_CHARS}]{{1,5}}(?:_[{_FACE_CHARS}]{{1,5}})*\)$")


def is_kaomoji(text: str) -> bool:
    """Heuristically detect kaomoji/emoticon-like tokens.

    This is deterministic and deliberately conservative. The main purpose is
    to avoid breaking kaomoji by transformations like underscore replacement.
    """
    s = text.strip()
    if not s:
        return False

    if len(s) > 24:
        return False

    # Exclude common "disambiguation tags" like `bow_(weapon)` or `ganyu_(genshin_impact)`.
    # These are standard tags, not kaomoji.
    if _DISAMBIGUATION.match(s):
        return False

    # Exclude path-like tags such as `/mlp/`.
    if _SLASH_TAG.match(s):
        return False

    if _SIMPLE_EMOTICON.match(s):
        return True

    if _COLON_ANGLE.match(s):
        return True

    # Parenthesized faces like (o_o), (9), etc. - check before _PARENS_WORD
    if _FACE_PARENS.match(s):
        return True

    # Exclude parenthesized word tags such as `(mlp)` but keep face-like ones handled above.
    if _PARENS_WORD.match(s):
        return False

    # Underscore-separated faces like ^_^, 0_0, >_<, 0_0_0, etc.
    if _FACE_UNDERSCORE.match(s):
        # Avoid false positives like `au_ra` by requiring at least one non-letter signal.
        if any(ch in s for ch in "^;:<>=") or any(ch.isdigit() for ch in s) or "o" in s.lower() or "x" in s.lower():
            return True

    alnum = 0
    punct = 0
    has_space = False
    for ch in s:
        if ch.isalnum():
            alnum += 1
        elif ch.isspace():
            has_space = True
        else:
            punct += 1

    # Final conservative heuristic fallback:
    # - must contain at least one typical face punctuation
    # - must be mostly non-alphanumeric
    if any(ch in s for ch in "^;:<>=") and (punct >= 2) and (alnum <= 3) and (not has_space):
        return True

    return False


def _escape_parentheses(text: str) -> str:
    # Match genai-tag-db-tools TagCleaner behavior: escape parentheses.
    return _UNESCAPED_LPAREN.sub(r"\\(", _UNESCAPED_RPAREN.sub(r"\\)", text))


def normalize_tag(source_tag: str) -> str:
    """入力CSVのsource_tagをTAGS.tagに変換する正規化関数.

    この関数はDB外のデータ変換のみに使用します。
    DB内のtag列は既に正規化済みです。

    Args:
        source_tag: 入力ソースからの生データ
            例: "spiked_collar", "Witch", ":D"

    Returns:
        正規化済みタグ
            例: "spiked collar", "witch", ":D"

    Examples:
        >>> normalize_tag("spiked_collar")
        'spiked collar'
        >>> normalize_tag("Witch")
        'witch'
        >>> normalize_tag("1_girl")
        '1 girl'
        >>> normalize_tag(":D")
        ':D'
    """
    s = source_tag.strip()
    if not s:
        return ""

    # 顔文字判定を先に行う（lowercase前の文字列で判定）
    if is_kaomoji(s):
        # 顔文字は正規化から除外（括弧エスケープのみ）
        return _escape_parentheses(s)

    # 通常タグは lowercase + アンダースコア置換
    s = s.lower()
    s = s.replace("_", " ")
    s = s.strip()
    return _escape_parentheses(s)
