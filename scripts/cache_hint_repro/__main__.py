"""Run the same source checkout or a downloaded Python zip application."""

import argparse
import sys
import tempfile
import zipfile
from pathlib import Path


def run_from(directory, args):
    sys.path.insert(0, str(directory))
    from measure import run

    run(args, directory)


def main():
    parser = argparse.ArgumentParser(description="Reproduce the device-kernel timings in FlagTree PR #1658.")
    ops = parser.add_mutually_exclusive_group()
    ops.add_argument("--op", default="abs", help="Public operator name (default: abs).")
    ops.add_argument("--all", action="store_true", help="Run every reported operator on the current platform.")
    parser.add_argument("--list", action="store_true", help="List the available operators on the current platform.")
    parser.add_argument("--cold-round", type=int, choices=(1, 2, 3), default=3,
                        help="Cold protocol to repeat. The PR latency columns use round 3.")
    parser.add_argument("--json", type=Path, metavar="FILE", help="Save timings, samples and executable checks.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if root.is_dir():
        run_from(root, args)
    else:
        # Triton's JIT needs a real source path for the small cache-sweep kernel.
        with tempfile.TemporaryDirectory(prefix="flagtree-cache-hint-") as directory:
            with zipfile.ZipFile(root) as package:
                package.extractall(directory)
            run_from(Path(directory), args)


if __name__ == "__main__":
    main()
