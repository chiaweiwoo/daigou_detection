"""Run the full pipeline end-to-end."""

import sys
from probuyer_xai.pipeline import run_all

if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
