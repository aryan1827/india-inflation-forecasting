"""Download every series that can be fetched automatically (FRED), and tell
you which ones still need a manual export from RBI DBIE / MoSPI.

    python scripts/fetch_data.py            # keep existing files
    python scripts/fetch_data.py --force    # re-download FRED series

Files you place in data/raw/ yourself are never overwritten unless --force.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from inflation_fc import config as C  # noqa: E402
from inflation_fc.data import download_missing  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-download FRED series even if files exist")
    args = ap.parse_args()
    print("Checking data/raw/ ...")
    status = download_missing(force=args.force)
    print()
    for name, st in status.items():
        print(f"  {name:7s} {st}")
    todo = [n for n, st in status.items() if st.startswith("missing") or "failed" in st]
    if status.get("cpi", "").startswith("downloaded"):
        print("\nNOTE: CPI came from FRED (OECD series, historically CPI for Industrial Workers).")
        print("      For the real project, replace data/raw/cpi_combined_index.csv with MoSPI's")
        print("      CPI-Combined all-India general index (see data/SOURCES.md).")
    if todo:
        print("\nStill needed (see data/SOURCES.md for exact steps):")
        for n in todo:
            spec = C.SERIES[n]
            flag = "required" if spec["required"] else "optional"
            print(f"  - {spec['file']}: {spec['label']} [{flag}] from {spec['source']}")
    else:
        print("\nAll series present. Next: python scripts/run_all.py")


if __name__ == "__main__":
    main()
