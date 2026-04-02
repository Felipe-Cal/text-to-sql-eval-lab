"""Seed the DuckDB database. Run once before evaluations."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.db import seed_database

if __name__ == "__main__":
    force = "--force" in sys.argv
    seed_database(force=force)
