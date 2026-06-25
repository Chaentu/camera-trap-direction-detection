"""pathusage -- automated direction and activity detection for camera-trap sequences."""

from .config import load_config
from .pipeline import process_site, run

__all__ = ["load_config", "process_site", "run"]
__version__ = "1.0.0"
