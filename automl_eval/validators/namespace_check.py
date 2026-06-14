""""""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from automl_eval.validators.base import BaseValidator, ValidationResult

if TYPE_CHECKING:
    from automl_eval.core.session import RuntimeSession

EXPECTED_ARTIFACTS = [
    {
        "name": "model",
        "alt_names": ["clf", "classifier", "regressor", "estimator", "pipeline"],
        "check": "has_predict",
        "description": "trained model with .predict()",
        "required_after_action": "MODEL",
    },
]


def _find_var(
    ns: dict[str, Any], name: str, alt_names: list[str]
) -> tuple[str | None, Any]:
    """Find a variable by primary name or alternatives."""
    if name in ns:
        return name, ns[name]
    for alt in alt_names:
        if alt in ns:
            return alt, ns[alt]
    return None, None


class NamespaceCheckValidator(BaseValidator):
    """"""

    name = "namespace_check"

    def validate(self, session: RuntimeSession) -> ValidationResult:
        from automl_eval.core.session import ActionType

        ns = session.sandbox_namespace
        issues: list[str] = []
        checks_passed = 0
        checks_total = 0

        past_actions = {s.action_type for s in session.steps}

        for artifact in EXPECTED_ARTIFACTS:
            required_action = artifact.get("required_after_action")
            if required_action:
                if ActionType(required_action) not in past_actions:
                    continue

            checks_total += 1
            var_name, var_val = _find_var(
                ns, artifact["name"], artifact.get("alt_names", [])
            )

            if var_val is None:
                issues.append(f"Missing: {artifact['description']}")
                continue

            check = artifact.get("check")
            if check == "has_predict" and not hasattr(var_val, "predict"):
                issues.append(
                    f"Variable '{var_name}' exists but has no .predict() method"
                )
                continue

            checks_passed += 1

        if checks_total == 0:
            return ValidationResult(
                validator_name=self.name,
                passed=True,
                score=1.0,
                details="No namespace checks applicable at this step.",
            )

        score = checks_passed / checks_total
        passed = len(issues) == 0

        return ValidationResult(
            validator_name=self.name,
            passed=passed,
            score=score,
            details="; ".join(issues) if issues else "All namespace checks passed.",
            penalty=0.1 * len(issues),
        )
