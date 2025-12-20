"""タグ正規化（source_tag → TAGS.tag）.

入力タグを、DBで扱う正規化済みタグ（TAGS.tag）に変換するための関数群です。

設計方針:
    - 正規化は「入力 → TAGS.tag」の変換に限定する（DB内のtagを再正規化しない）
    - 顔文字（短いエモーティコン）は壊れやすいため、保守的に判定して通常正規化から除外する
    - 括弧は genai-tag-db-tools の TagCleaner 互換のためエスケープする
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

# 短いASCII顔文字/エモーティコンでよく使われる文字
_FACE_CHARS = r"0-9oOxXtTuUvV\.\-\+;:\^<>=_"
_FACE_UNDERSCORE = re.compile(rf"^[{_FACE_CHARS}]{{1,3}}(?:_[{_FACE_CHARS}]{{1,3}})+$")
_FACE_PARENS = re.compile(rf"^\([{_FACE_CHARS}]{{1,5}}(?:_[{_FACE_CHARS}]{{1,5}})*\)$")


def is_kaomoji(text: str) -> bool:
    """顔文字/エモーティコンっぽいトークンを（保守的に）判定する.

    ここは決定的（deterministic）かつ保守的な判定に寄せています。
    目的は「顔文字を通常タグとして扱って _ → space 置換などで壊す」事故を避けることです。
    """
    s = text.strip()
    if not s:
        return False

    if len(s) > 24:
        return False

    # `bow_(weapon)` のような曖昧さ回避タグは通常タグなので除外
    if _DISAMBIGUATION.match(s):
        return False

    # `/mlp/` のようなパス風タグは除外
    if _SLASH_TAG.match(s):
        return False

    if _SIMPLE_EMOTICON.match(s):
        return True

    if _COLON_ANGLE.match(s):
        return True

    # (o_o) のような括弧付き顔文字（単語括弧より先に判定）
    if _FACE_PARENS.match(s):
        return True

    # `(mlp)` のような括弧付き単語タグは除外（顔文字は上で拾っている）
    if _PARENS_WORD.match(s):
        return False

    # ^_^ / 0_0 / >_< などのアンダースコア区切り顔文字
    if _FACE_UNDERSCORE.match(s) and (
        any(ch in s for ch in "^;:<>=")
        or any(ch.isdigit() for ch in s)
        or "o" in s.lower()
        or "x" in s.lower()
    ):
        # `au_ra` のような誤検出を避けるため、記号/数字などのシグナルを要求
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

    # 最終フォールバック（かなり保守的）
    # - 顔文字っぽい記号を含む
    # - 非英数字が多い
    return any(ch in s for ch in "^;:<>=") and (punct >= 2) and (alnum <= 3) and (not has_space)


def _escape_parentheses(text: str) -> str:
    """括弧をエスケープする（genai-tag-db-tools互換）."""
    return _UNESCAPED_LPAREN.sub(r"\\(", _UNESCAPED_RPAREN.sub(r"\\)", text))


def canonicalize_source_tag(source_tag: str) -> str:
    """source_tag（入力ソースの代表表記）をDB保存向けに整形する.

    方針（会話で確定した内容）:
    - `source_tag` は代表1つのみ保持し、case揺れはノイズなので小文字に統一する
    - ただし顔文字は小文字化で壊れるため、そのまま保持する
    - `source_tag` 側ではアンダースコア置換や括弧エスケープは行わない（元表記を残す）
      - `_ -> space` と括弧エスケープは TAGS.tag 側（`normalize_tag`）の責務
    """
    s = source_tag.strip()
    if not s:
        return ""

    if is_kaomoji(s):
        return s

    return s.lower()


def normalize_tag(source_tag: str) -> str:
    """入力の source_tag を TAGS.tag に変換する正規化関数.

    Args:
        source_tag: 入力ソースからの生タグ（例: "spiked_collar", "Witch", ":D"）

    Returns:
        正規化済みタグ（例: "spiked collar", "witch", ":D"）

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

    # 顔文字は lowercase/アンダースコア置換をすると壊れるので、先に除外する
    if is_kaomoji(s):
        # 顔文字は正規化から除外（括弧エスケープのみ）
        return _escape_parentheses(s)

    # 通常タグは lowercase + アンダースコア置換
    #
    # NOTE:
    # - 区切り記号として全角コロン等が混ざっているケースがあるため、ここで半角へ寄せる
    # - 顔文字は上で除外済みなので、ここでの置換が壊すリスクは小さい
    fullwidth_to_ascii = str.maketrans(
        {
            "：": ":",
            "，": ",",
            "．": ".",
            "；": ";",
            "！": "!",
            "？": "?",
            "（": "(",
            "）": ")",
            "［": "[",
            "］": "]",
            "｛": "{",
            "｝": "}",
            "／": "/",
            "＼": "\\\\",
            "＋": "+",
            "＝": "=",
            "－": "-",
            "＊": "*",
            "＆": "&",
            "％": "%",
            "＃": "#",
            "＠": "@",
            "＾": "^",
            "～": "~",
            "｜": "|",
            "＜": "<",
            "＞": ">",
            "　": " ",  # 全角スペース
        }
    )
    s = s.translate(fullwidth_to_ascii)
    s = s.lower()
    s = s.replace("_", " ")
    s = s.strip()
    return _escape_parentheses(s)
