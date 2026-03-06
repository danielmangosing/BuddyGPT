"""CLI wrapper for the BuddyGPT Windows smoke harness."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.windows_smoke import main


if __name__ == "__main__":
    raise SystemExit(main())
