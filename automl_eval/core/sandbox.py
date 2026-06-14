"""Restricted execution for agent code with policy errors and actionable repair hints."""

from __future__ import annotations

import ast
import io
import signal
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from typing import Any

FORBIDDEN_MODULES = frozenset(
    {
        "os",
        "subprocess",
        "shutil",
        "pathlib",
        "glob",
        "socket",
        "http",
        "urllib",
        "requests",
        "httpx",
        "sys",
        "importlib",
        "ctypes",
        "pickle",
        "joblib",
    }
)
FORBIDDEN_FUNCTIONS = frozenset(
    {"open", "exec", "eval", "compile", "__import__", "input"}
)
FORBIDDEN_METHODS = frozenset(
    {
        "read_csv",
        "read_excel",
        "read_json",
        "read_parquet",
        "read_pickle",
        "read_feather",
        "read_hdf",
        "read_sql",
        "to_csv",
        "to_excel",
        "to_json",
        "to_parquet",
        "to_pickle",
        "listdir",
        "walk",
        "scandir",
        "system",
        "popen",
        "getenv",
        "environ",
        "install",
    }
)
VALIDATION_METRIC_CALLS = frozenset(
    {
        "roc_auc_score",
        "accuracy_score",
        "f1_score",
        "log_loss",
        "mean_squared_error",
        "root_mean_squared_error",
        "mean_absolute_error",
        "r2_score",
        "classification_report",
        "confusion_matrix",
        "balanced_accuracy_score",
        "average_precision_score",
    }
)


@dataclass
class ExecutionResult:
    success: bool
    stdout: str
    stderr: str
    error: str | None = None
    returned_value: Any = None


class TimeoutError(Exception):
    pass


class SandboxPolicyError(Exception):
    pass


def _timeout_handler(signum: int, frame: Any) -> None:
    raise TimeoutError("Code execution timed out")


_original_import = __import__


def _safe_import(name: str, *args: Any, **kwargs: Any) -> Any:
    root = name.split(".", 1)[0]
    if root in FORBIDDEN_MODULES:
        raise ImportError(f"Import of '{name}' is forbidden in the sandbox.")
    return _original_import(name, *args, **kwargs)


_BUILTINS_TO_REMOVE = (FORBIDDEN_FUNCTIONS - {"__import__"}) | {"breakpoint"}


def _sanitized_builtins() -> dict[str, Any]:
    """Return a copy of the host builtins with exfiltration-capable names removed."""
    base = (
        dict(__builtins__)
        if isinstance(__builtins__, dict)
        else vars(__builtins__).copy()
    )
    for name in _BUILTINS_TO_REMOVE:
        base.pop(name, None)
    base["__import__"] = _safe_import
    return base


def _qualified_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        left = _qualified_name(node.value)
        return f"{left}.{node.attr}" if left else node.attr
    return ""


def _call_mentions_names(node: ast.AST, suspicious_names: set[str]) -> bool:
    """Return True if a metric call appears to touch evaluator-owned data."""
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id.lower() in suspicious_names:
            return True
        if isinstance(child, ast.Attribute):
            q = _qualified_name(child).lower()
            if any(name in q for name in suspicious_names):
                return True
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            v = child.value.lower()
            if any(name in v for name in suspicious_names):
                return True
    return False


def _metric_call_policy_issue(node: ast.Call, leaf: str) -> str | None:
    suspicious_names = {
        "valid_df",
        "validation_df",
        "val_df",
        "x_valid",
        "x_val",
        "y_valid",
        "y_val",
        "valid_y",
        "validation_y",
        "private_dev_df",
        "hidden_test_df",
        "test_df",
        "x_test",
        "y_test",
    }
    if _call_mentions_names(node, suspicious_names):
        return (
            f"call to `{leaf}` appears to score evaluator-owned validation/test data; "
            "register a pipeline/predict_fn and select ACTION: VALIDATE instead"
        )
    return None


def _validate_policy(code: str, *, allow_validation_metrics: bool) -> None:
    """Reject data exfiltration and metric bypass attempts before code can run."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return
    issues: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".", 1)[0] in FORBIDDEN_MODULES:
                    issues.append(f"import `{alias.name}` is not permitted")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".", 1)[0] in FORBIDDEN_MODULES:
                issues.append(f"import from `{node.module}` is not permitted")
        elif isinstance(node, ast.Call):
            name = _qualified_name(node.func)
            leaf = name.rsplit(".", 1)[-1]
            if leaf in FORBIDDEN_FUNCTIONS or leaf in FORBIDDEN_METHODS:
                issues.append(f"call to `{name or leaf}` is not permitted")
            if not allow_validation_metrics and leaf in VALIDATION_METRIC_CALLS:
                issue = _metric_call_policy_issue(node, leaf)
                if issue:
                    issues.append(issue)
    if issues:
        unique = list(dict.fromkeys(issues))
        raise SandboxPolicyError("; ".join(unique))


def _error_hint(exc: Exception, namespace: dict[str, Any]) -> str:
    """Return concrete runtime repair guidance without exposing hidden scoring logic."""
    msg = str(exc).lower()
    if isinstance(exc, SandboxPolicyError):
        return (
            "Use only in-memory sandbox variables and evaluator-owned scoring. "
            "Manual sklearn.metrics calls are allowed only for train-local diagnostics; do not score "
            "valid_df/private_dev_df/hidden-test rows yourself. Register a replayable `pipeline` or "
            "`predict_fn`, then request evaluator metrics with ACTION: VALIDATE."
        )
    if isinstance(exc, FileNotFoundError) or "read_csv" in msg:
        return (
            "Do NOT read files from disk. Data is already loaded in memory as `train_df` and `valid_df`; "
            "the test split is not exposed."
        )
    if isinstance(exc, NameError):
        name = str(exc).split("'")[1] if "'" in str(exc) else ""
        suggestion = {
            "train": "train_df",
            "valid": "valid_df",
            "df_train": "train_df",
            "df": "train_df",
        }.get(name)
        if suggestion:
            return f"Variable '{name}' is not provided. Use `{suggestion}` directly."
        if name and name in _SKLEARN_NAME_TO_IMPORT_LINE:
            import_line = _SKLEARN_NAME_TO_IMPORT_LINE[name]
            return (
                f"`{name}` is not imported in this code block. Add at the top of your code:\n"
                f"    {import_line}\n"
                "Only `pd` (pandas) and `np` (numpy) are pre-bound. If a previous turn "
                "appeared to import this name, that turn likely failed -- when a turn "
                "fails, all of its bindings (including imports made before the failing "
                "line) are rolled back atomically. Re-import what you need at the top "
                "of THIS code block; this is the safest pattern regardless of "
                "previous-turn namespace state."
            )
    if isinstance(exc, KeyError):
        raw = str(exc)
        missing_cols = _parse_keyerror_missing_columns(raw)
        if missing_cols:
            engineered_hits: list[tuple[str, str]] = []
            for col in missing_cols:
                in_locals = _columns_in_locals(namespace, col)
                if in_locals:
                    engineered_hits.append((col, in_locals[0]))
            if engineered_hits:
                if len(engineered_hits) == 1:
                    col, frame = engineered_hits[0]
                    return (
                        f"Column `{col}` exists in local variable `{frame}` but NOT in `train_df` or "
                        "`train_df_original`. The ColumnTransformer cannot read it from raw rows at predict time. "
                        "Wrap your feature-engineering function inside `FunctionTransformer(feature_engineering, "
                        "validate=False)` and place it as the FIRST step of your `Pipeline`, before the "
                        "ColumnTransformer. Then the engineered columns will be created from raw rows automatically. "
                        "Remember to also import `FunctionTransformer` from `sklearn.preprocessing`."
                    )
                cols_str = ", ".join(f"`{c}`" for c, _ in engineered_hits)
                frame = engineered_hits[0][1]
                return (
                    f"Engineered columns {cols_str} exist in local variable `{frame}` but NOT in `train_df` or "
                    "`train_df_original`. The ColumnTransformer / DataFrame indexing cannot read them from raw rows "
                    "at predict time. Wrap your feature-engineering function in "
                    "`FunctionTransformer(feature_engineering, validate=False)` and place it as the FIRST step of "
                    "your `Pipeline`, before the ColumnTransformer. Then the engineered columns will be created from "
                    "raw rows automatically. Remember to also import `FunctionTransformer` from `sklearn.preprocessing`."
                )
            if len(missing_cols) == 1:
                missing = missing_cols[0]
                working = namespace.get("train_df")
                protected = namespace.get("train_df_original")
                if hasattr(working, "columns") and hasattr(protected, "columns"):
                    if missing not in working.columns and missing in protected.columns:
                        return (
                            f"Column `{missing}` was removed from the mutable working frame. "
                            "Rebuild local features from `train_df_original.copy()` or place the transformation inside "
                            "a raw-input pipeline/predict_fn; do not modify the protected snapshot."
                        )
                return f"Column `{missing}` is not available in the dataframe being used; inspect its columns before transforming."
            cols_str = ", ".join(f"`{c}`" for c in missing_cols)
            return (
                f"Columns {cols_str} are not available in the dataframe being used; inspect its columns "
                "before transforming. If you engineered them in an earlier step, wrap that function inside "
                "`FunctionTransformer(...)` and put it at the start of your Pipeline so the columns are "
                "created from raw rows automatically."
            )
        missing = raw.strip("'\"")
        return f"Column `{missing}` is not available in the dataframe being used; inspect its columns before transforming."
    if "notfittederror" in type(exc).__name__.lower() or "not fitted" in msg:
        return (
            "Fit imputers/encoders only on training data, preferably inside one sklearn Pipeline, "
            "then let that fitted pipeline transform unseen rows."
        )
    if (
        isinstance(exc, ValueError)
        and "cannot use median strategy with non-numeric data" in msg
    ):
        return (
            "A median imputer was applied to non-numeric columns. Use a ColumnTransformer inside the submitted "
            "raw-input Pipeline: numeric columns -> SimpleImputer(strategy='median'); categorical/object columns -> "
            "SimpleImputer(strategy='most_frequent') followed by OneHotEncoder(handle_unknown='ignore')."
        )
    if isinstance(exc, ValueError) and "could not convert string to float" in msg:
        return (
            "Raw categorical columns reached a numeric estimator. Encode or drop categoricals inside a "
            "ColumnTransformer that is part of the submitted raw-input pipeline. A robust default is: "
            "numeric pipeline with median imputation; categorical pipeline with most_frequent imputation + "
            "OneHotEncoder(handle_unknown='ignore')."
        )
    if isinstance(exc, ValueError) and (
        "contains nan" in msg or "input contains nan" in msg
    ):
        return (
            "Target or feature values contain NaN. If the target column contains nulls in raw rows, "
            "drop those rows before fitting (e.g. `y = train_df[target_column]; mask = y.notna()`). "
            "For feature NaNs, route them through a SimpleImputer / KNNImputer inside your Pipeline rather "
            "than imputing in a separate workspace step that does not get replayed at predict time."
        )
    if isinstance(exc, AttributeError) and "predict_proba" in msg:
        return (
            "The final estimator has no `predict_proba` method (it is likely a regressor or a non-probabilistic "
            "classifier). For probability metrics (ROC-AUC, log-loss), use a probabilistic classifier such as "
            "LogisticRegression, RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, "
            "or any sklearn classifier that implements `predict_proba`."
        )
    if isinstance(exc, ValueError) and "not a column of the dataframe" in msg:
        return (
            "A ColumnTransformer references a column that does not exist in the input DataFrame at "
            "fit/transform time. If you engineered the column in an earlier step, place that engineering "
            "inside the submitted Pipeline -- e.g. as a `FunctionTransformer(your_feature_fn)` first step -- "
            "so the column is created from raw rows automatically when scoring."
        )
    if isinstance(exc, ValueError) and ("columns" in msg or "feature names" in msg):
        return (
            "Training and unseen raw-row transformations are inconsistent. Encapsulate feature creation and "
            "encoding inside `predict_fn(raw_dataframe)` or one raw-input Pipeline."
        )
    return ""


_SKLEARN_NAME_TO_IMPORT_LINE: dict[str, str] = {
    # Pipeline / composition
    "Pipeline": "from sklearn.pipeline import Pipeline",
    "FeatureUnion": "from sklearn.pipeline import FeatureUnion",
    "make_pipeline": "from sklearn.pipeline import make_pipeline",
    "ColumnTransformer": "from sklearn.compose import ColumnTransformer",
    "make_column_transformer": "from sklearn.compose import make_column_transformer",
    "TransformedTargetRegressor": "from sklearn.compose import TransformedTargetRegressor",
    # Preprocessing
    "FunctionTransformer": "from sklearn.preprocessing import FunctionTransformer",
    "StandardScaler": "from sklearn.preprocessing import StandardScaler",
    "MinMaxScaler": "from sklearn.preprocessing import MinMaxScaler",
    "RobustScaler": "from sklearn.preprocessing import RobustScaler",
    "MaxAbsScaler": "from sklearn.preprocessing import MaxAbsScaler",
    "PowerTransformer": "from sklearn.preprocessing import PowerTransformer",
    "QuantileTransformer": "from sklearn.preprocessing import QuantileTransformer",
    "Normalizer": "from sklearn.preprocessing import Normalizer",
    "OneHotEncoder": "from sklearn.preprocessing import OneHotEncoder",
    "OrdinalEncoder": "from sklearn.preprocessing import OrdinalEncoder",
    "LabelEncoder": "from sklearn.preprocessing import LabelEncoder",
    "TargetEncoder": "from sklearn.preprocessing import TargetEncoder",
    "KBinsDiscretizer": "from sklearn.preprocessing import KBinsDiscretizer",
    "PolynomialFeatures": "from sklearn.preprocessing import PolynomialFeatures",
    "Binarizer": "from sklearn.preprocessing import Binarizer",
    # Imputation
    "SimpleImputer": "from sklearn.impute import SimpleImputer",
    "KNNImputer": "from sklearn.impute import KNNImputer",
    "IterativeImputer": "from sklearn.experimental import enable_iterative_imputer\nfrom sklearn.impute import IterativeImputer",
    "MissingIndicator": "from sklearn.impute import MissingIndicator",
    # Linear models
    "LogisticRegression": "from sklearn.linear_model import LogisticRegression",
    "LinearRegression": "from sklearn.linear_model import LinearRegression",
    "Ridge": "from sklearn.linear_model import Ridge",
    "RidgeClassifier": "from sklearn.linear_model import RidgeClassifier",
    "Lasso": "from sklearn.linear_model import Lasso",
    "ElasticNet": "from sklearn.linear_model import ElasticNet",
    "SGDClassifier": "from sklearn.linear_model import SGDClassifier",
    "SGDRegressor": "from sklearn.linear_model import SGDRegressor",
    # Tree ensembles
    "RandomForestClassifier": "from sklearn.ensemble import RandomForestClassifier",
    "RandomForestRegressor": "from sklearn.ensemble import RandomForestRegressor",
    "GradientBoostingClassifier": "from sklearn.ensemble import GradientBoostingClassifier",
    "GradientBoostingRegressor": "from sklearn.ensemble import GradientBoostingRegressor",
    "HistGradientBoostingClassifier": "from sklearn.ensemble import HistGradientBoostingClassifier",
    "HistGradientBoostingRegressor": "from sklearn.ensemble import HistGradientBoostingRegressor",
    "ExtraTreesClassifier": "from sklearn.ensemble import ExtraTreesClassifier",
    "ExtraTreesRegressor": "from sklearn.ensemble import ExtraTreesRegressor",
    "BaggingClassifier": "from sklearn.ensemble import BaggingClassifier",
    "AdaBoostClassifier": "from sklearn.ensemble import AdaBoostClassifier",
    "VotingClassifier": "from sklearn.ensemble import VotingClassifier",
    "StackingClassifier": "from sklearn.ensemble import StackingClassifier",
    "IsolationForest": "from sklearn.ensemble import IsolationForest",
    # Trees
    "DecisionTreeClassifier": "from sklearn.tree import DecisionTreeClassifier",
    "DecisionTreeRegressor": "from sklearn.tree import DecisionTreeRegressor",
    # Neighbors, SVM, naive Bayes, neural nets
    "KNeighborsClassifier": "from sklearn.neighbors import KNeighborsClassifier",
    "KNeighborsRegressor": "from sklearn.neighbors import KNeighborsRegressor",
    "SVC": "from sklearn.svm import SVC",
    "LinearSVC": "from sklearn.svm import LinearSVC",
    "SVR": "from sklearn.svm import SVR",
    "GaussianNB": "from sklearn.naive_bayes import GaussianNB",
    "BernoulliNB": "from sklearn.naive_bayes import BernoulliNB",
    "MultinomialNB": "from sklearn.naive_bayes import MultinomialNB",
    "MLPClassifier": "from sklearn.neural_network import MLPClassifier",
    "MLPRegressor": "from sklearn.neural_network import MLPRegressor",
    # Model selection (do NOT include score-revealing helpers like cross_val_score)
    "train_test_split": "from sklearn.model_selection import train_test_split",
    "StratifiedKFold": "from sklearn.model_selection import StratifiedKFold",
    "KFold": "from sklearn.model_selection import KFold",
    "GridSearchCV": "from sklearn.model_selection import GridSearchCV",
    "RandomizedSearchCV": "from sklearn.model_selection import RandomizedSearchCV",
    # Decomposition / feature selection
    "PCA": "from sklearn.decomposition import PCA",
    "TruncatedSVD": "from sklearn.decomposition import TruncatedSVD",
    "SelectKBest": "from sklearn.feature_selection import SelectKBest",
    "SelectFromModel": "from sklearn.feature_selection import SelectFromModel",
    "VarianceThreshold": "from sklearn.feature_selection import VarianceThreshold",
    # Base mixins (rare but seen in pilots)
    "BaseEstimator": "from sklearn.base import BaseEstimator",
    "TransformerMixin": "from sklearn.base import TransformerMixin",
    "ClassifierMixin": "from sklearn.base import ClassifierMixin",
    "RegressorMixin": "from sklearn.base import RegressorMixin",
    "clone": "from sklearn.base import clone",
    # scipy and numpy/pandas commonly forgotten
    "stats": "from scipy import stats",
    "sparse": "from scipy import sparse",
}


def _parse_keyerror_missing_columns(raw: str) -> list[str]:
    """Normalise a stringified ``KeyError`` to a list of missing column names."""
    text = raw.strip()
    # Strip a single layer of surrounding quotes (the ``"'..."'`` shape).
    if (text.startswith("'") and text.endswith("'")) or (
        text.startswith('"') and text.endswith('"')
    ):
        text = text[1:-1].strip()
    import re as _re

    match = _re.match(r"\[(.+)\]\s*not in index", text)
    if match:
        inner = match.group(1)
        parts = [p.strip().strip("'\"") for p in inner.split(",")]
        return [p for p in parts if p]
    match = _re.match(r"\[(.+)\]\s*not (?:in|found in) (?:index|axis)", text)
    if match:
        inner = match.group(1)
        parts = [p.strip().strip("'\"") for p in inner.split(",")]
        return [p for p in parts if p]
    return [text] if text else []


def _columns_in_locals(namespace: dict[str, Any], missing: str) -> list[str]:
    """Return names of local DataFrames in the agent namespace that contain ``missing``."""
    import pandas as _pd

    candidates: list[str] = []
    protected_names = {"train_df", "valid_df", "train_df_original", "valid_df_original"}
    for name, value in namespace.items():
        if name in protected_names or name.startswith("__"):
            continue
        if isinstance(value, _pd.DataFrame) and missing in value.columns:
            candidates.append(name)
    return candidates


class Sandbox:
    """Agent code executor with offline/policy restrictions and stdout capture."""

    def __init__(self, timeout_seconds: int = 60) -> None:
        self.timeout_seconds = timeout_seconds

    def execute(
        self,
        code: str,
        namespace: dict[str, Any],
        *,
        allow_validation_metrics: bool = False,
    ) -> ExecutionResult:
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        safe_builtins = _sanitized_builtins()
        namespace["__builtins__"] = safe_builtins
        old_handler = None
        try:
            _validate_policy(code, allow_validation_metrics=allow_validation_metrics)
            if hasattr(signal, "SIGALRM"):
                old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(self.timeout_seconds)
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(compile(code, "<agent_code>", "exec"), namespace)
            if hasattr(signal, "SIGALRM"):
                signal.alarm(0)
            return ExecutionResult(True, stdout_buf.getvalue(), stderr_buf.getvalue())
        except TimeoutError:
            return ExecutionResult(
                False,
                stdout_buf.getvalue(),
                stderr_buf.getvalue(),
                error=f"Execution timed out after {self.timeout_seconds}s",
            )
        except Exception as exc:
            if hasattr(signal, "SIGALRM"):
                signal.alarm(0)
            tb = traceback.format_exc()
            hint = _error_hint(exc, namespace)
            error_msg = f"{type(exc).__name__}: {exc}\n{tb}"
            if hint:
                error_msg += f"\n\nHINT: {hint}"
            return ExecutionResult(
                False, stdout_buf.getvalue(), stderr_buf.getvalue(), error=error_msg
            )
        finally:
            if old_handler is not None and hasattr(signal, "SIGALRM"):
                signal.signal(signal.SIGALRM, old_handler)
