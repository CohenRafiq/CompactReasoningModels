#!/usr/bin/env python3
"""
Reshape the raw input array for a nonogram and move any leading zeros
to the trailing side of each clue row.

Usage:
    python remake_inputs.py <input_path> <nonogram_size> [--output-dir DIR] [--suffix SUFFIX]

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


def shift_zeros_to_trail(arr: np.ndarray) -> np.ndarray:
    """
    For each innermost row (the clue list) move all zeros to the right while
    preserving the order of the non‑zero entries.
    """
    # ``arr == 0`` creates a boolean mask; a stable argsort puts ``False`` (non‑zero)
    # before ``True`` (zero) while keeping the original order of the non‑zeros.
    idx = np.argsort(arr == 0, axis=-1, kind="stable")
    return np.take_along_axis(arr, idx, axis=-1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reshape inputs and shift leading zeros to the end."
    )
    parser.add_argument("input_path", help="Path to the raw inputs file (.npy or .npz)")
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

    src = Path(args.input_path)
    raw = load_array(src)

    # ------------------------------------------------------------------
    # 1️⃣ Reshape to (N, 2, nonogram_size, (nonogram_size+1)//2)
    # ------------------------------------------------------------------
    n_samples = raw.shape[0]
    x_shape = (2, args.nonogram_size, (args.nonogram_size + 1) // 2)
    inputs = raw.reshape((n_samples,) + x_shape)

    # ------------------------------------------------------------------
    # 2️⃣ Shift leading zeros → trailing zeros (per clue row)
    # ------------------------------------------------------------------
    inputs = shift_zeros_to_trail(inputs)

    # ------------------------------------------------------------------
    # 3️⃣ Save as .npy
    # ------------------------------------------------------------------
    out_dir = Path(args.output_dir) if args.output_dir else src.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{src.stem}{args.suffix}.npy"
    save_npy(inputs, out_path)

    print(f"✓ Inputs reshaped & saved → {out_path}  shape={inputs.shape}")


if __name__ == "__main__":
    main()