#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_experiments.sh [--config PATH] [--smoke] [runner args...]

Examples:
  bash scripts/run_experiments.sh
  bash scripts/run_experiments.sh --smoke --dry-run
  bash scripts/run_experiments.sh --config configs/final_paper.yaml --resume
  bash scripts/run_experiments.sh --config configs/final_paper.yaml --resume --rerun-transient
  bash scripts/run_experiments.sh --workers 4 --temperatures 0.2,0.7

Notes:
  - Uses uv and forwards remaining flags to `python -m automl_eval.experiment.run_grace_experiments`.
  - Common forwarded flags: --resume, --rerun-transient, --dry-run, --workers,
    --models, --tasks, --regimes, --temperatures, --output, --run-name.
  - Full runs require OPENROUTER_API_KEY unless --dry-run is passed through.
EOF
}

die() {
  printf '[run_experiments][error] %s\n' "$*" >&2
  exit 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ "${1-}" == "-h" || "${1-}" == "--help" ]]; then
  usage
  exit 0
fi

command -v uv >/dev/null 2>&1 || die '`uv` is required but was not found in PATH.'

cd "$REPO_ROOT"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
exec uv run python -m automl_eval.experiment.run_grace_experiments "$@"
