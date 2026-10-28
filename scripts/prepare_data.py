"""Download real FHFA data if absent, clean, engineer features, and split by time."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import Config
from src.prepare import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(ROOT / Config.raw_path))
    parser.add_argument("--out", default=str(ROOT / "data/processed"))
    parser.add_argument("--first-test-year", type=int, default=Config.first_test_year)
    args = parser.parse_args()
    if not Path(args.data).exists():
        subprocess.run([sys.executable, str(ROOT / "scripts/download_data.py"),
                        "--out", args.data], check=True)
    prepare(Config(raw_path=args.data, first_test_year=args.first_test_year), args.out)


if __name__ == "__main__":
    main()
