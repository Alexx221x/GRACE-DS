""""""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from automl_eval.core.session import RuntimeSession


class ValidatorStatus(str, Enum):
    INACTIVE = "inactive"
    UNRESOLVED = "unresolved"
    IMPROVED = "improved"
    RESOLVED = "resolved"
    REGRESSED = "regressed"
    BLOCKED = "blocked"


@dataclass
class ValidationResult:
    """Result of a single validator."""

    validator_name: str
    passed: bool
    score: float
    details: str = ""
    penalty: float = 0.0
    status: ValidatorStatus = ValidatorStatus.UNRESOLVED


class BaseValidator(ABC):
    """Validator interface: accepts a session, returns ValidationResult."""

    name: str = "base"

    @abstractmethod
    def validate(self, session: RuntimeSession) -> ValidationResult: ...
