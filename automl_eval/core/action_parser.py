"""Parse strict stage-labelled agent responses and fenced executable code."""

from __future__ import annotations

import re
from dataclasses import dataclass

from automl_eval.core.session import ActionType
from automl_eval.core.think_filter import strip_reasoning


@dataclass
class ParsedAction:
    action_type: ActionType
    body: str
    raw_text: str
    code_block_count: int = 0

    @property
    def has_code_block(self) -> bool:
        return self.code_block_count > 0


_ACTION_HEADER = re.compile(
    r"^\s*ACTION\s*:\s*(PLAN|EDA|FEATURE_ENGINEERING|MODEL|VALIDATE|CODE|CODE_FIX|FINAL_SUBMIT)\s*$",
    re.MULTILINE | re.IGNORECASE,
)
_FENCED_BLOCK = re.compile(
    r"```([^`\n\r]*)\s*(?:\r?\n)?(.*?)```", re.DOTALL | re.IGNORECASE
)
_EXECUTABLE_FENCE_LANGS = {"", "python", "py"}


def _executable_code_blocks(text: str) -> list[str]:
    """Return fenced blocks that are safe to treat as Python code."""
    blocks: list[str] = []
    for lang, body in _FENCED_BLOCK.findall(text):
        norm = (lang or "").strip().lower()
        # Some models emit `````python extra````; take the first token as the language.
        first = norm.split()[0] if norm else ""
        if first in _EXECUTABLE_FENCE_LANGS:
            blocks.append(body.strip())
    return blocks


_HEURISTIC_KEYWORDS: dict[ActionType, list[str]] = {
    ActionType.PLAN: ["plan:", "objective:", "strategy:", "approach:", "steps:"],
    ActionType.EDA: [
        "describe(",
        ".info(",
        "corr(",
        "isna(",
        "missing",
        "value_counts",
        "explor",
    ],
    ActionType.FEATURE_ENGINEERING: [
        "standardscaler",
        "onehotencod",
        "fillna",
        "imputer",
        "transform",
        "feature",
    ],
    ActionType.MODEL: [
        "lightgbm",
        "xgboost",
        "randomforest",
        "logisticregression",
        ".fit(",
        "gridsearch",
    ],
    ActionType.VALIDATE: ["validate", "validation metric", "evaluate", "score("],
    ActionType.CODE_FIX: ["fix", "исправ", "ошибк", "error", "traceback", "bug"],
    ActionType.FINAL_SUBMIT: ["final_submit", "submit", "predict_fn", "финальн"],
}


class ActionParser:
    """Keep the stage text for feedback but execute only explicit fenced Python."""

    def parse(self, text: str) -> ParsedAction:
        text = strip_reasoning(text).strip()
        match = _ACTION_HEADER.search(text)
        if match:
            action_type = ActionType(match.group(1).upper())
            raw_body = text[match.end() :].strip()
            blocks = _executable_code_blocks(raw_body)
            body = blocks[0].strip() if len(blocks) == 1 else raw_body
            return ParsedAction(action_type, body, text, len(blocks))
        blocks = _executable_code_blocks(text)
        if blocks:
            return ParsedAction(ActionType.CODE, blocks[0].strip(), text, len(blocks))
        return ParsedAction(self._guess_type(text), text, text, 0)

    def _guess_type(self, text: str) -> ActionType:
        lower = text.lower()
        scores = {kind: 0 for kind in ActionType}
        for kind, keywords in _HEURISTIC_KEYWORDS.items():
            scores[kind] = sum(keyword in lower for keyword in keywords)
        winner = max(scores, key=scores.get)
        return winner if scores[winner] > 0 else ActionType.CODE
