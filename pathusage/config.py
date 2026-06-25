"""Configuration loading with sensible defaults."""

from __future__ import annotations

import json
from pathlib import Path

# Default parameters. Anything in config.json overrides these.
DEFAULTS = {
    "output_dir": "results",
    "daytime_hours": None,        # e.g. [6, 20] to keep only 06:00-19:59; null = all hours
    "sequence_gap_seconds": 120,  # a new sequence starts after a gap longer than this
    "detect_conf": 0.20,          # detection confidence threshold
    "count_conf": 0.40,           # confidence threshold for activity counting
    "grid_cols": 5,               # virtual-grid columns along the path axis
    "grid_rows": 3,               # virtual-grid rows (recorded, not used for direction)
    "min_move_px": 15.0,          # minimum centroid displacement for the trajectory fallback
    "speed_slow_px_s": 80,        # speed hint threshold: below -> slow
    "speed_fast_px_s": 300,       # speed hint threshold: above -> fast
}

# Default per-site calibration if a site omits it.
DEFAULT_CALIB = {
    "axis": "x",        # "x" if movement is left/right in the image, "y" if up/down
    "in_sign": "+",     # "+" if the inbound/uphill direction is increasing along `axis`
    "in_label": "in",   # label written for the inbound direction
    "out_label": "out", # label written for the outbound direction
}


def load_config(path="config.json"):
    """Load config.json and fill in any missing values with the defaults."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}. Copy config.example.json to "
            f"config.json and edit it.")
    cfg = json.loads(path.read_text(encoding="utf-8"))

    for key, value in DEFAULTS.items():
        cfg.setdefault(key, value)

    if "sites" not in cfg or not cfg["sites"]:
        raise ValueError("config.json must define at least one entry under 'sites'.")

    for site, site_cfg in cfg["sites"].items():
        if "image_folders" not in site_cfg or not site_cfg["image_folders"]:
            raise ValueError(f"Site '{site}' must define a non-empty 'image_folders' list.")
        calib = dict(DEFAULT_CALIB)
        calib.update(site_cfg.get("calibration", {}))
        site_cfg["calibration"] = calib

    return cfg
