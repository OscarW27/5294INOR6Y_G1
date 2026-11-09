"""Download the Endomondo HR prediction dataset from Hugging Face.

Usage (from the repo root):
    python scripts/download_data.py            # all files (~680 MB)
    python scripts/download_data.py --splits   # only train/validation/test (~340 MB)

Files are saved to data/endomondo-hr-prediction-v2/. Files that are already
present and complete are skipped, so it is safe to re-run.
"""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "rricc22/endomondo-hr-prediction-v2"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "endomondo-hr-prediction-v2"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--splits",
        action="store_true",
        help="only download train/validation/test, not the combined dataset.parquet",
    )
    args = parser.parse_args()

    patterns = ["README.md", "train.parquet", "validation.parquet", "test.parquet"]
    if not args.splits:
        patterns.append("dataset.parquet")

    print(f"Downloading {REPO_ID} to {DATA_DIR} ...")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=DATA_DIR,
        allow_patterns=patterns,
    )

    print("\nDone. Files:")
    for f in sorted(DATA_DIR.glob("*")):
        if f.is_file():
            print(f"  {f.name:<22} {f.stat().st_size / 1e6:8.1f} MB")


if __name__ == "__main__":
    main()
