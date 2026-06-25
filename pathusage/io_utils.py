"""Image loading utilities: EXIF timestamps, deduplicated listing,
sequence grouping, and image-size lookup."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PIL import Image, ExifTags

EXIF_IDS = {v: k for k, v in ExifTags.TAGS.items()}
IMG_EXTS = {".jpg", ".jpeg"}


def get_exif_datetime(path):
    """Read the capture time from a JPEG EXIF header.

    Falls back to the file-modification time if no parseable EXIF
    timestamp is present.
    """
    path = Path(path)
    try:
        with Image.open(path) as im:
            exif = im.getexif()
            sub = exif.get_ifd(ExifTags.IFD.Exif)  # DateTimeOriginal lives in the Exif sub-IFD
            for source in (sub, exif):
                for tag in ("DateTimeOriginal", "DateTimeDigitized", "DateTime"):
                    tid = EXIF_IDS.get(tag)
                    if tid and tid in source:
                        val = source[tid]
                        if isinstance(val, bytes):
                            val = val.decode(errors="ignore")
                        return datetime.strptime(val.strip(), "%Y:%m:%d %H:%M:%S")
    except Exception:
        pass
    return datetime.fromtimestamp(path.stat().st_mtime)


def get_image_size(path):
    """Return (width, height) of an image in pixels."""
    with Image.open(path) as im:
        return im.size  # (width, height)


def list_images(folders, daytime_hours=None):
    """List images from one or more folders, sorted by capture time.

    Deduplication: images with the same (timestamp, filename) are treated
    as duplicates and loaded only once. This handles camera filename resets
    across deployments. If ``daytime_hours`` is given as [start, end], only
    images captured in that hour range are kept.
    """
    items = {}
    for folder in folders:
        folder = Path(folder)
        if not folder.exists():
            print(f"  WARNING: folder not found: {folder}")
            continue
        for p in folder.iterdir():
            if p.suffix.lower() not in IMG_EXTS:
                continue
            dt = get_exif_datetime(p)
            if daytime_hours and not (daytime_hours[0] <= dt.hour < daytime_hours[1]):
                continue
            key = (dt, p.name)
            if key not in items:
                items[key] = {"path": p, "dt": dt}
    return sorted(items.values(), key=lambda x: (x["dt"], x["path"].name))


def group_sequences(items, gap_seconds=120.0):
    """Group time-sorted images into movement sequences.

    A new sequence begins when the gap to the previous image exceeds
    ``gap_seconds``. Within a sequence, images keep their capture order so
    that the first and last frames define the temporal endpoints used for
    direction and speed estimation.
    """
    if not items:
        return []
    sequences, current = [], [items[0]]
    for item in items[1:]:
        if (item["dt"] - current[-1]["dt"]).total_seconds() > gap_seconds:
            sequences.append(current)
            current = []
        current.append(item)
    sequences.append(current)
    return sequences
