"""End-to-end pipeline: detection -> sequences -> activity + direction -> Excel.

For each configured site the pipeline lists the images, groups them into
movement sequences, runs the detector (cached on disk), classifies the activity
and estimates the direction per sequence, and writes one Excel (and CSV) file
with one row per sequence.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from .classification import classify_sequence
from .detection import Detector
from .direction import estimate_direction
from .io_utils import get_image_size, group_sequences, list_images

COLUMN_ORDER = [
    "timestamp", "location_id", "sequence_id", "primary_activity",
    "n_total_persons", "n_mtb", "n_on_foot", "n_dogs", "n_horses",
    "n_other_wildlife", "n_empty_frames", "direction", "speed_px_s",
    "activity_hint", "confidence_flag", "n_images", "n_points",
]


def _build_cache(sequences, detector, frame_w, frame_h):
    """Run detection on every image and cache the results per sequence."""
    cache = []
    for seq in sequences:
        seq_dets = [detector.detect(item["path"]) for item in seq]
        cache.append({
            "timestamp": seq[0]["dt"],
            "n_images": len(seq),
            "dt_offsets": [(it["dt"] - seq[0]["dt"]).total_seconds() for it in seq],
            "dets": seq_dets,
            "frame_w": frame_w,
            "frame_h": frame_h,
        })
    return cache


def process_site(site_name, site_cfg, cfg, detector=None):
    """Process one site and write its Excel / CSV output. Returns the DataFrame."""
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    calib = site_cfg["calibration"]

    print(f"\n=== Site: {site_name} ===")
    images = list_images(site_cfg["image_folders"], cfg["daytime_hours"])
    if not images:
        print("  No images found -- skipping.")
        return None
    sequences = group_sequences(images, cfg["sequence_gap_seconds"])
    frame_w, frame_h = get_image_size(images[0]["path"])
    print(f"  {len(images)} images -> {len(sequences)} sequences "
          f"(frame {frame_w}x{frame_h})")

    # Detection (cached on disk; delete the .pkl to re-detect)
    cache_path = out_dir / f"{site_name}_cache.pkl"
    if cache_path.exists():
        cache = pickle.load(open(cache_path, "rb"))
        print(f"  Loaded detection cache ({len(cache)} sequences). "
              f"Delete {cache_path.name} to re-detect.")
    else:
        if detector is None:
            detector = Detector(conf=cfg["detect_conf"])
        print("  Running detection (this can take a while on CPU)...")
        cache = _build_cache(sequences, detector, frame_w, frame_h)
        pickle.dump(cache, open(cache_path, "wb"))
        print(f"  Saved detection cache -> {cache_path.name}")

    # Classification + direction per sequence
    records = []
    for i, entry in enumerate(cache):
        cls = classify_sequence(entry["dets"], cfg["count_conf"])
        dir_res = estimate_direction(
            entry["dets"], entry["dt_offsets"], calib,
            entry["frame_w"], entry["frame_h"],
            detect_conf=cfg["detect_conf"], n_cols=cfg["grid_cols"],
            min_move_px=cfg["min_move_px"],
            speed_slow=cfg["speed_slow_px_s"], speed_fast=cfg["speed_fast_px_s"])

        # No detected person -> no direction
        if cls["n_total_persons"] == 0:
            dir_res["direction"] = "undefined"

        rec = {**cls, **dir_res,
               "timestamp": entry["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
               "location_id": site_name,
               "sequence_id": i,
               "n_images": entry["n_images"],
               "confidence_flag": "low" if dir_res.get("n_points", 0) < 2 else "ok"}
        records.append(rec)

    df = pd.DataFrame(records)
    df = df[[c for c in COLUMN_ORDER if c in df.columns]]

    xlsx = out_dir / f"{site_name}_sequences.xlsx"
    csv = out_dir / f"{site_name}_sequences.csv"
    df.to_excel(xlsx, index=False)
    df.to_csv(csv, index=False)
    print(f"  Saved {len(df)} sequences -> {xlsx.name}")

    defined = (df["direction"] != "undefined").mean() * 100 if len(df) else 0
    print(f"  Activity: {df['primary_activity'].value_counts().to_dict()}")
    print(f"  Direction defined: {defined:.1f}%  "
          f"({df['direction'].value_counts().to_dict()})")
    return df


def run(cfg):
    """Process every site defined in the config."""
    detector = None
    for site_name, site_cfg in cfg["sites"].items():
        # Reuse one detector instance across sites (loaded lazily on first need)
        if detector is None and not (Path(cfg["output_dir"]) /
                                     f"{site_name}_cache.pkl").exists():
            detector = Detector(conf=cfg["detect_conf"])
        process_site(site_name, site_cfg, cfg, detector)
