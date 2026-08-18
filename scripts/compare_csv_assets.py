"""Compare generated CSV experiment assets against committed baselines."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path


def _numeric(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def compare_file(expected: Path, actual: Path, rtol: float, atol: float) -> list[str]:
    errors: list[str] = []
    with expected.open(encoding="utf-8", newline="") as left, actual.open(
        encoding="utf-8", newline=""
    ) as right:
        expected_rows = list(csv.reader(left))
        actual_rows = list(csv.reader(right))

    if len(expected_rows) != len(actual_rows):
        return [
            f"{actual.name}: row count {len(actual_rows)} != {len(expected_rows)}"
        ]

    paired_rows = zip(expected_rows, actual_rows, strict=True)
    for i, (exp_row, act_row) in enumerate(paired_rows):
        if len(exp_row) != len(act_row):
            errors.append(
                f"{actual.name}: row {i} column count {len(act_row)} != {len(exp_row)}"
            )
            continue
        for j, (exp, act) in enumerate(zip(exp_row, act_row, strict=True)):
            exp_num = _numeric(exp)
            act_num = _numeric(act)
            if exp_num is not None and act_num is not None:
                if math.isnan(exp_num) and math.isnan(act_num):
                    continue
                if not math.isclose(exp_num, act_num, rel_tol=rtol, abs_tol=atol):
                    errors.append(
                        f"{actual.name}: row {i}, column {j}: "
                        f"{act} != {exp} within tolerance"
                    )
            elif exp != act:
                errors.append(
                    f"{actual.name}: row {i}, column {j}: {act!r} != {exp!r}"
                )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("expected", type=Path)
    parser.add_argument("actual", type=Path)
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--atol", type=float, default=1e-10)
    args = parser.parse_args()

    errors: list[str] = []
    for expected in sorted(args.expected.glob("*.csv")):
        actual = args.actual / expected.name
        if not actual.exists():
            errors.append(f"missing generated asset: {actual}")
            continue
        errors.extend(compare_file(expected, actual, args.rtol, args.atol))

    if errors:
        print("\n".join(errors[:50]), file=sys.stderr)
        if len(errors) > 50:
            print(f"... {len(errors) - 50} additional mismatches", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
