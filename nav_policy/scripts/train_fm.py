"""Thin CLI wrapper — delegates to nav_policy.train.train_fm.main()."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nav_policy.train.train_fm import main

if __name__ == "__main__":
    main()
