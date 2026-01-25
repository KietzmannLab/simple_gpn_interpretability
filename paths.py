# paths.py
"""
Centralized, repo-safe paths for outputs/artifacts/caches.

Design goals:
- Portable (no absolute lab paths)
- Deterministic defaults (repo-relative)
- User-overridable via environment variables
- Creates directories on import (safe + convenient)

Env overrides:
- RUNS_DIR
- MODELS_DIR
- SEQUENCES_DIR
- ARTIFACTS_DIR
- OUTPUTS_DIR
"""

from __future__ import annotations
from pathlib import Path
import os


def _repo_root() -> Path:
    # This file should live in the repo (top-level or inside src/).
    # If you place it under src/, change parents[1] -> parents[2] accordingly.
    return Path(__file__).resolve().parent


REPO_ROOT: Path = _repo_root()

# Base folders (overrideable)
OUTPUTS_DIR: Path = Path(os.getenv("OUTPUTS_DIR", REPO_ROOT / "outputs")).resolve()
ARTIFACTS_DIR: Path = Path(os.getenv("ARTIFACTS_DIR", REPO_ROOT / "artifacts")).resolve()

# Specific folders (overrideable)
MODELS_DIR: Path = Path(os.getenv("MODELS_DIR", ARTIFACTS_DIR / "models")).resolve()
SEQUENCES_DIR: Path = Path(os.getenv("SEQUENCES_DIR", ARTIFACTS_DIR / "sequences")).resolve()

# Ensure dirs exist
for _p in (OUTPUTS_DIR, ARTIFACTS_DIR, MODELS_DIR, SEQUENCES_DIR):
    _p.mkdir(parents=True, exist_ok=True)