"""Shared helpers for the diagnostic scripts.

Run scripts from anywhere, e.g.:
    python scripts/01_schema_check.py
Each script adds this directory to sys.path so `from _common import ...` works.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import yaml
except ImportError:
    sys.exit("Missing dependency 'pyyaml'. Run: pip install -r requirements.txt")

# Repo root = parent of the scripts/ directory that holds this file.
REPO_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | None = None) -> dict:
    cfg_path = Path(path) if path else REPO_ROOT / "config" / "config.yaml"
    if not cfg_path.exists():
        sys.exit(f"Config not found at {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve(p: str) -> Path:
    """Resolve a (possibly relative) config path against the repo root."""
    pp = Path(p)
    return pp if pp.is_absolute() else (REPO_ROOT / pp)


def ensure_dirs(cfg: dict) -> None:
    for key in ("processed_dir", "tables_dir", "figures_dir", "predictions_dir", "logs_dir"):
        d = cfg["paths"].get(key)
        if d:
            resolve(d).mkdir(parents=True, exist_ok=True)


def load_raw_panel(cfg: dict, verbose: bool = True) -> pd.DataFrame:
    """Read the .RData file and return the firm-year panel.

    Handles the quirk that the file may contain several objects (or an object
    whose internal name differs from the file name): we pick the data frame
    that has both the firm-id and year columns and the most rows.
    """
    try:
        import pyreadr
    except ImportError:
        sys.exit("Missing dependency 'pyreadr'. Run: pip install -r requirements.txt")

    raw_path = resolve(cfg["paths"]["raw_data"])
    if not raw_path.exists():
        sys.exit(f"Raw data not found at {raw_path}\n"
                 f"Put iesCompleto.RData in {raw_path.parent} and rerun.")

    if verbose:
        print(f"[load] reading {raw_path}  (large .RData can take ~1 min)...")
    result = pyreadr.read_r(str(raw_path))  # OrderedDict: {object_name: DataFrame}
    if verbose:
        print(f"[load] objects inside the file: {list(result.keys())}")

    fid, yr = cfg["columns"]["firm_id"], cfg["columns"]["year"]
    matches = [(nm, obj) for nm, obj in result.items()
               if isinstance(obj, pd.DataFrame) and fid in obj.columns and yr in obj.columns]
    if matches:
        nm, df = max(matches, key=lambda t: t[1].shape[0])
        if verbose:
            print(f"[load] using object '{nm}' as the panel  shape={df.shape}")
        return df.copy()

    frames = [(nm, obj) for nm, obj in result.items() if isinstance(obj, pd.DataFrame)]
    if not frames:
        sys.exit("No data frame found inside the .RData file.")
    nm, df = max(frames, key=lambda t: t[1].shape[0] * t[1].shape[1])
    print(f"[load] WARNING no object had both '{fid}' and '{yr}'. Using largest object '{nm}'.")
    return df.copy()


def standardise_ids(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    fid, yr = cfg["columns"]["firm_id"], cfg["columns"]["year"]
    df[fid] = df[fid].astype("string").str.strip()
    df[yr] = pd.to_numeric(df[yr], errors="coerce").astype("Int64")
    return df


def to_numeric(df: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    for c in names:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def safe_divide(num, den) -> np.ndarray:
    """Elementwise num/den, returning NaN where den is 0 or non-finite."""
    num = np.asarray(num, dtype="float64")
    den = np.asarray(den, dtype="float64")
    out = np.full(np.broadcast(num, den).shape, np.nan, dtype="float64")
    ok = np.isfinite(den) & (den != 0) & np.isfinite(num)
    out[ok] = num[ok] / den[ok]
    return out


def derive_division(cae_series: pd.Series) -> pd.Series:
    """2-digit NACE division from a CAE code (digits only, first two)."""
    digits = cae_series.astype("string").str.replace(r"\D", "", regex=True)
    return pd.to_numeric(digits.str[:2], errors="coerce").astype("Int64")
