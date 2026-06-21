"""Download UCI Online Retail xlsx to data/raw/."""

import sys
from probuyer_xai.data import download_raw
from probuyer_xai.config import RAW_XLSX

if __name__ == "__main__":
    success = download_raw()
    if not success:
        print()
        print("=" * 60)
        print("ACTION REQUIRED")
        print("=" * 60)
        print("Please download the UCI Online Retail dataset manually:")
        print("  URL: https://archive.ics.uci.edu/ml/datasets/Online+Retail")
        print(f"  Save to: {RAW_XLSX}")
        print("Then re-run this script or scripts/06_run_all.py")
        print("=" * 60)
        sys.exit(1)
