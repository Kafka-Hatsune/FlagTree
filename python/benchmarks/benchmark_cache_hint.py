# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Benchmark runner for FlagTree PR #1658 cold-start and warm-start cache-hint evaluation."""

from pathlib import Path
import runpy
import sys


def main():
    repro_dir = Path(__file__).resolve().parents[2] / "scripts" / "cache_hint_repro"
    if str(repro_dir) not in sys.path:
        sys.path.insert(0, str(repro_dir))
    runpy.run_path(str(repro_dir / "__main__.py"), run_name="__main__")


if __name__ == "__main__":
    main()
