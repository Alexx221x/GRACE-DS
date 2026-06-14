"""Public runtime facts included in agent prompts for reproducible paper runs."""

from __future__ import annotations

from importlib import metadata

APPROVED_LIBRARY_DISTRIBUTIONS: tuple[tuple[str, str], ...] = (
    ("Python", "python"),
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("scikit-learn", "scikit-learn"),
    ("scipy", "scipy"),
    ("statsmodels", "statsmodels"),
)


def approved_library_versions() -> dict[str, str]:
    """Return evaluator-approved public version strings for prompt display."""
    import platform

    versions: dict[str, str] = {"Python": platform.python_version()}
    for display_name, distribution_name in APPROVED_LIBRARY_DISTRIBUTIONS[1:]:
        try:
            versions[display_name] = metadata.version(distribution_name)
        except metadata.PackageNotFoundError:
            versions[display_name] = "not installed"
    return versions


def approved_library_versions_text() -> str:
    """Format public runtime versions for insertion in system/observation prompts."""
    return ", ".join(
        f"{name}={version}" for name, version in approved_library_versions().items()
    )


PRINT_FEEDBACK_INSTRUCTION = (
    "Python execution feedback includes exceptions and values explicitly written to stdout. "
    "To inspect computed results from EDA or debugging, use print(...) in your Python code; "
    "an unprinted final expression is not guaranteed to be shown in the next observation."
)
