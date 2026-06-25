#!/usr/bin/env python3
"""Command-line entry point.

Usage:
    python run_pipeline.py                 # process all sites in config.json
    python run_pipeline.py --site my_site  # process one site
    python run_pipeline.py --config other_config.json
"""

import argparse

from pathusage import load_config
from pathusage.pipeline import process_site, run


def main():
    ap = argparse.ArgumentParser(description="Camera-trap direction & activity pipeline")
    ap.add_argument("--config", default="config.json", help="path to config file")
    ap.add_argument("--site", default=None, help="process only this site")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.site:
        if args.site not in cfg["sites"]:
            raise SystemExit(f"Site '{args.site}' not found in {args.config}.")
        process_site(args.site, cfg["sites"][args.site], cfg)
    else:
        run(cfg)
    print("\nDone.")


if __name__ == "__main__":
    main()
