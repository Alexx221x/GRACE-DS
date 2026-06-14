#!/usr/bin/env bash
# =============================================================================
# GRACE — one-command data installer
#
# Clones + prepares the two external dataset repos and materialises every GRACE
# task (TML-bench + TabReD + synthetic) into automl_eval/tasks/.
#
# Dependency isolation: TML-bench and TabReD pin different numpy/pandas versions
# than GRACE, so each is prepared inside its OWN throwaway virtualenv. GRACE's
# environment is never modified; it only READS the produced files at the end.
#
# Usage (from the GRACE repo root, with GRACE deps already installed):
#     bash scripts/install_data.sh
#
# Common options:
#     --data-dir DIR        where to clone/prepare external repos (default: ./external)
#     --tml-root DIR        use an EXISTING prepared tml-bench clone (skip clone+prep)
#     --tabred-root DIR     use an EXISTING prepared tabred clone (skip clone+prep)
#     --tabred-subset "a,b" only prepare these TabReD datasets
#                           (default: homesite,sberbank-housing; upstream
#                           ships 6 others: ecom-offers, homecredit,
#                           cooking-time, delivery-eta, maps-routing, weather)
#     --max-rows N          subsample large TabReD datasets (default 60000; 0 = all)
#     --download-timeout-sec N
#                           timeout for internet-backed preparation steps
#                           (default: $GRACE_DATA_DOWNLOAD_TIMEOUT_SEC or 14400)
#     --skip-tml            do not prepare TML-bench
#     --skip-tabred         do not prepare TabReD
#     --skip-synthetic      do not build the synthetic tasks
#     --no-reference        skip baseline/oracle reference-score computation
#
# Requires: git, python3 (3.11+), and Kaggle credentials for the real datasets
# (~/.kaggle/kaggle.json, chmod 600, OR export KAGGLE_CONFIG_DIR=/path/to/dir).
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRACE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
GRACE_PY="$(command -v python3)"

# --- defaults ---
DATA_DIR="$GRACE_ROOT/external"
TML_ROOT=""
TABRED_ROOT=""
TABRED_SUBSET=""
MAX_ROWS="60000"
DO_TML=1
DO_TABRED=1
DO_SYNTHETIC=1
REFERENCE_FLAG=""
TML_REPO="https://github.com/mykolapinchuk/tml-bench.git"
TABRED_REPO="https://github.com/yandex-research/tabred.git"
TML_COMPS=(playground-series-s6e1 playground-series-s5e10 bank-customer-churn-ict-u-ai foot-traffic-wuerzburg-retail-forecasting-2-0)
TABRED_SCRIPTS=(sberbank-housing)
DATA_DOWNLOAD_TIMEOUT_SEC="${GRACE_DATA_DOWNLOAD_TIMEOUT_SEC:-14400}"

# --- parse args ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir) DATA_DIR="$2"; shift 2;;
    --tml-root) TML_ROOT="$2"; shift 2;;
    --tabred-root) TABRED_ROOT="$2"; shift 2;;
    --tabred-subset) TABRED_SUBSET="$2"; shift 2;;
    --max-rows) MAX_ROWS="$2"; shift 2;;
    --download-timeout-sec) DATA_DOWNLOAD_TIMEOUT_SEC="$2"; shift 2;;
    --skip-tml) DO_TML=0; shift;;
    --skip-tabred) DO_TABRED=0; shift;;
    --skip-synthetic) DO_SYNTHETIC=0; shift;;
    --no-reference) REFERENCE_FLAG="--no-reference"; shift;;
    -h|--help) sed -n '2,40p' "$0"; exit 0;;
    *) echo "Unknown option: $1" >&2; exit 1;;
  esac
done

log() { printf '\n\033[1;36m[install_data] %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[install_data][warn] %s\033[0m\n' "$*" >&2; }
run_with_download_timeout() {
  if command -v timeout >/dev/null 2>&1; then
    timeout "${DATA_DOWNLOAD_TIMEOUT_SEC}" "$@"
  else
    warn "GNU timeout command not found; running without an enforced download timeout."
    "$@"
  fi
}

mkdir -p "$DATA_DIR"

# --- preflight: GRACE deps + Kaggle creds ---
log "GRACE python: $GRACE_PY"
log "Internet dataset preparation timeout: ${DATA_DOWNLOAD_TIMEOUT_SEC}s"
( cd "$GRACE_ROOT" && "$GRACE_PY" -c "import pandas, numpy, yaml, pyarrow" ) 2>/dev/null \
  || warn "GRACE deps incomplete in $GRACE_PY. Run: pip install -r requirements.txt -r requirements-experiments.txt"

if [[ $DO_TML -eq 1 || $DO_TABRED -eq 1 ]]; then
  if [[ -z "${KAGGLE_CONFIG_DIR:-}" && ! -f "$HOME/.kaggle/kaggle.json" ]]; then
    warn "No Kaggle credentials found (~/.kaggle/kaggle.json or \$KAGGLE_CONFIG_DIR). Downloads will fail."
    warn "Get a token at https://www.kaggle.com/settings -> Create New Token, and accept each competition's rules."
  fi
fi

make_venv() {  # $1 = venv path
  if [[ ! -d "$1" ]]; then
    "$GRACE_PY" -m venv "$1"
    "$1/bin/python" -m pip install --upgrade pip -q
  fi
}

# =============================================================================
# 1) TML-bench
# =============================================================================
if [[ $DO_TML -eq 1 ]]; then
  if [[ -z "$TML_ROOT" ]]; then
    TML_ROOT="$DATA_DIR/tml-bench"
    if [[ ! -d "$TML_ROOT/.git" && ! -f "$TML_ROOT/README.md" ]]; then
      log "Cloning TML-bench -> $TML_ROOT"
      git clone --depth 1 "$TML_REPO" "$TML_ROOT"
    else
      log "Using existing TML-bench clone: $TML_ROOT"
    fi
    TML_VENV="$DATA_DIR/.venv_tml"
    log "Creating TML-bench prep venv: $TML_VENV"
    make_venv "$TML_VENV"
    "$TML_VENV/bin/pip" install -q -r "$TML_ROOT/requirements.txt"
    TML_PY="$TML_VENV/bin/python"
    for c in "${TML_COMPS[@]}"; do
      if [[ -f "$TML_ROOT/competitions/$c/public/train_public.csv" ]]; then
        log "TML-bench: $c already prepared, skipping download."
      else
        log "TML-bench: preparing $c (Kaggle download) ..."
        ( cd "$TML_ROOT" && run_with_download_timeout "$TML_PY" "competitions/$c/prepare_competition.py" --download )
      fi
    done
  else
    log "Using pre-prepared TML-bench at: $TML_ROOT (no clone/prep)"
  fi
fi

# =============================================================================
# 2) TabReD
# =============================================================================
if [[ $DO_TABRED -eq 1 ]]; then
  if [[ -z "$TABRED_ROOT" ]]; then
    TABRED_ROOT="$DATA_DIR/tabred"
    if [[ ! -d "$TABRED_ROOT/.git" && ! -f "$TABRED_ROOT/readme.md" ]]; then
      log "Cloning TabReD -> $TABRED_ROOT"
      git clone --depth 1 "$TABRED_REPO" "$TABRED_ROOT"
    else
      log "Using existing TabReD clone: $TABRED_ROOT"
    fi
    TABRED_VENV="$DATA_DIR/.venv_tabred"
    log "Creating TabReD prep venv: $TABRED_VENV (installs CPU torch; this is the slow step)"
    make_venv "$TABRED_VENV"
    "$TABRED_VENV/bin/pip" install -q \
      "numpy==1.26.4" "pandas==2.2.1" "polars==0.20.19" "pyarrow==15.0.2" \
      "scikit-learn==1.4.1.post1" "scipy==1.13.0" "kaggle==1.6.11" "loguru==0.7.2" \
      "plotnine==0.13.4" "rtdl_num_embeddings==0.0.9" "rtdl_revisiting_models==0.0.2" \
      "xlsx2csv==0.8.1" "delu==0.0.23" \
      "tomli==2.0.1" "tomli-w==1.0" "tqdm==4.66.2" matplotlib "category_encoders==2.6.3" \
      torch --extra-index-url https://download.pytorch.org/whl/cpu
    ( cd "$TABRED_ROOT" && "$TABRED_VENV/bin/pip" install -q -e . )
    TABRED_PY="$TABRED_VENV/bin/python"
    mkdir -p "$TABRED_ROOT/data"
    if [[ -z "${KAGGLE_CONFIG_DIR:-}" && ! -f "$HOME/.kaggle/kaggle.json" ]]; then
      warn "Kaggle credentials missing. Create ~/.kaggle/kaggle.json with chmod 600 OR export KAGGLE_CONFIG_DIR=/path/to/dir."
    fi
    declare -A _TABRED_COMP_URL=(
      [homesite]="https://www.kaggle.com/competitions/homesite-quote-conversion"
      [sberbank-housing]="https://www.kaggle.com/competitions/sberbank-russian-housing-market"
      [homecredit]="https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability"
      [ecom-offers]="https://www.kaggle.com/c/acquire-valued-shoppers-challenge"
    )
    # Determine which scripts to run.
    RUN_SCRIPTS=("${TABRED_SCRIPTS[@]}")
    if [[ -n "$TABRED_SUBSET" ]]; then
      IFS=',' read -ra WANT <<< "$TABRED_SUBSET"
      RUN_SCRIPTS=("${WANT[@]}")
    fi
    TABRED_LOG_DIR="$DATA_DIR/.tabred_logs"
    mkdir -p "$TABRED_LOG_DIR"
    TABRED_FAILED=()
    for s in "${RUN_SCRIPTS[@]}"; do
      s="$(echo "$s" | xargs)"  # trim
      log "TabReD: preparing $s ..."
      ( cd "$TABRED_ROOT" && "$TABRED_PY" "preprocessing/$s.py" ) \
        || warn "TabReD $s failed (large datasets need lots of RAM/disk; see preprocessing/README.md)."
    done
  else
    log "Using pre-prepared TabReD at: $TABRED_ROOT (no clone/prep)"
  fi
fi

# =============================================================================
# 3) Materialise GRACE tasks (in GRACE's own environment)
# =============================================================================
log "Materialising GRACE tasks with $GRACE_PY ..."
PREP_ARGS=()
[[ $DO_TML -eq 1 ]] && PREP_ARGS+=(--tml-root "$TML_ROOT")
[[ $DO_TABRED -eq 1 ]] && PREP_ARGS+=(--tabred-root "$TABRED_ROOT")
[[ $DO_SYNTHETIC -eq 1 ]] && PREP_ARGS+=(--synthetic)
[[ -n "$TABRED_SUBSET" ]] && PREP_ARGS+=(--tabred-only "$TABRED_SUBSET")
[[ -n "$REFERENCE_FLAG" ]] && PREP_ARGS+=("$REFERENCE_FLAG")
PREP_ARGS+=(--max-rows "$MAX_ROWS")

( cd "$GRACE_ROOT" && "$GRACE_PY" -m automl_eval.dataset_loaders.prepare_datasets "${PREP_ARGS[@]}" )
