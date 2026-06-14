from __future__ import annotations

import json
from pathlib import Path

__version__ = "3.4.1"
PAPER_EXPERIMENTS_TAG = "v3.3-paper-experiments"


def version_info() -> dict:
    """Return the parsed VERSION.json shipped at the repository root."""
    here = Path(__file__).resolve().parent.parent
    vp = here / "VERSION.json"
    if vp.exists():
        return json.loads(vp.read_text(encoding="utf-8"))
    return {
        "release_version": __version__,
        "paper_experiments_tag": PAPER_EXPERIMENTS_TAG,
    }
