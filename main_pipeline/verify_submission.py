"""Verify the exact submitted bytes: python -m main_pipeline.verify_submission [CSV]."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .config import Paths, REPO


def verify_reference(path: Path) -> str:
    reference = json.loads((REPO / "reproduction.json").read_text())["submission"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != reference["sha256"]:
        raise RuntimeError(
            f"{path}: SHA-256 {digest} differs from the submitted file "
            f"({reference['sha256']}). See the reference GPU environment in reproduction.json."
        )
    print(f"Exact submission verified: {path}\nSHA-256 {digest}")
    return digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", type=Path, default=Paths.from_env().submission)
    args = parser.parse_args()
    verify_reference(args.csv)


if __name__ == "__main__":
    main()
