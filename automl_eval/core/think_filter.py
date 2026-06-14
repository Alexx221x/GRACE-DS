"""Strip chain-of-thought / reasoning blocks from LLM responses"""

from __future__ import annotations

import re


_REASONING_TAGS = ("think", "thinking", "reason", "reasoning")

_OPEN = "|".join(_REASONING_TAGS)
_BLOCK_RE = re.compile(
    rf"<\s*(?:{_OPEN})\s*>.*?<\s*/\s*(?:{_OPEN})\s*>",
    re.DOTALL | re.IGNORECASE,
)
_DANGLING_OPEN_RE = re.compile(
    rf"<\s*(?:{_OPEN})\s*>.*\Z",
    re.DOTALL | re.IGNORECASE,
)
_TRAILING_CLOSE_RE = re.compile(
    rf"\A.*<\s*/\s*(?:{_OPEN})\s*>",
    re.DOTALL | re.IGNORECASE,
)


def strip_reasoning(text: str) -> str:
    """Return ``text`` with any chain-of-thought reasoning removed."""
    if not text:
        return text
    has_open = re.search(rf"<\s*(?:{_OPEN})\s*>", text, re.IGNORECASE)
    has_close = re.search(rf"<\s*/\s*(?:{_OPEN})\s*>", text, re.IGNORECASE)
    if not has_open and not has_close:
        return text

    cleaned = text
    cleaned = _BLOCK_RE.sub("", cleaned)

    if re.search(rf"<\s*/\s*(?:{_OPEN})\s*>", cleaned, re.IGNORECASE):
        cleaned = _TRAILING_CLOSE_RE.sub("", cleaned, count=1)

    if re.search(rf"<\s*(?:{_OPEN})\s*>", cleaned, re.IGNORECASE):
        cleaned = _DANGLING_OPEN_RE.sub("", cleaned, count=1)

    return cleaned.strip()
