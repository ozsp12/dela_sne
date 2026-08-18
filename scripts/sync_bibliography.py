"""Synchronize the canonical bibliography into the self-contained paper tree."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "references" / "references.bib"
PAPER_COPY = ROOT / "paper" / "references.bib"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail instead of writing when paper/references.bib is stale.",
    )
    args = parser.parse_args()

    canonical = CANONICAL.read_bytes()
    current = PAPER_COPY.read_bytes() if PAPER_COPY.exists() else b""
    if current == canonical:
        return 0
    if args.check:
        print(
            "paper/references.bib differs from references/references.bib; "
            "run python scripts/sync_bibliography.py",
            file=sys.stderr,
        )
        return 1
    PAPER_COPY.write_bytes(canonical)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
