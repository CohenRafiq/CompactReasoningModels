#!/usr/bin/env python3
"""
Reshape the raw target array for a nonogram.

Usage:
    python remake_targets.py <target_path> <nonogram_size> [--output-dir DIR] [--suffix SUFFIX]

The script always writes a *.npy* file (even if the source is *.npz*).
"""

from pathlib import Path
import numpy as np
import argparse


def load_array(p: Path) -> np.ndarray:
    """Load a .npy or .npz file and return the first array."""
    if p.suffix == ".npz":
        arch = np.load(p)
        key = list(arch.keys())[0]          # take the first (usually only) array
        return arch[key]
    return np.load(p)                       # plain .npy


def save_npy(arr: np.ndarray, dst: Path) -> None:
    """Save ``arr`` as a .npy file."""
    np.save(dst, arr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reshape targets for a nonogram."
    )
    parser.add_argument("target_path", help="Path to the raw targets file (.npy or .npz)")
    parser.add_argument(
        "nonogram_size",
        type=int,
        help="Side length of the nonogram grid (e.g. 5)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for the reshaped file (default: same as source)",
    )
    parser.add_argument(
        "--suffix",
        default="_reshaped",
        help="Suffix added before the .npy extension (default: _reshaped)",
    )
    args = parser.parse_args()

    src = Path(args.target_path)
    raw = load_array(src)

    # ------------------------------------------------------------------
    # 1️⃣ Reshape to (N, nonogram_size, nonogram_size)
    # ------------------------------------------------------------------
    n_samples = raw.shape[0]
    y_shape = (args.nonogram_size, args.nonogram_size)
    targets = raw.reshape((n_samples,) + y_shape)

    # ------------------------------------------------------------------
    # 2️⃣ Save as .npy
    # ------------------------------------------------------------------
    out_dir = Path(args.output_dir) if args.output_dir else src.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{src.stem}{args.suffix}.npy"
    save_npy(targets, out_path)

    print(f"✓ Targets reshaped & saved → {out_path}  shape={targets.shape}")


if __name__ == "__main__":
    main()