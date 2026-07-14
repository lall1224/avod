#!/usr/bin/env python3
"""End-to-end CPU smoke test on synthetic data."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYN = ROOT / "examples" / "synthetic"


def run(cmd):
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(ROOT))


def main():
    run([sys.executable, "examples/make_synthetic_data.py", "--out-dir", str(SYN)])
    run(
        [
            sys.executable,
            "scripts/train.py",
            "--annotations",
            str(SYN / "annotations.json"),
            "--split",
            str(SYN / "split.json"),
            "--features",
            str(SYN / "features"),
            "--model",
            "profile",
            "--epochs",
            "2",
            "--batch-size",
            "8",
            "--output-dir",
            str(SYN / "outputs"),
        ]
    )
    print("\nSmoke test OK.")


if __name__ == "__main__":
    main()
