import gc
import json
import multiprocessing as mp
import os
import time
from collections import defaultdict

import numpy as np
import torch

from compactreasoningmodels.datasets.collate import collate_combined
from compactreasoningmodels.datasets.nonogram_dataset import NonogramDataset
from compactreasoningmodels.solvers import SOLVERS

# TEMP_SOLVERS = {
#     "mac": SOLVERS["mac"],
#     "gradient_descent_global_sgd": SOLVERS["gradient_descent_global_sgd"],
#     "backtracking_search": SOLVERS["backtracking_search"],
# }
TEMP_SOLVERS = {
    "crl1": SOLVERS["gradient_descent_global_sgd"],
}

# SAMPLING_RATIO_RUNS = [(1.0, 1), (0.75, 3), (0.5, 3)]
SAMPLING_RATIO_RUNS = [(1.0, 1)]

# Per-worker solver instances (created once, reused across puzzles)
_worker_solvers: dict = {}


def to_serialisable(x):
    if torch.is_tensor(x):
        return x.tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if isinstance(x, (tuple, list)):
        return [to_serialisable(v) for v in x]
    if isinstance(x, dict):
        return {k: to_serialisable(v) for k, v in x.items()}
    return x


def _pool_init(solver_names: list[str]):
    """Create solver instances once per worker process."""
    global _worker_solvers
    seed = np.random.randint(0, 2**31)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    _worker_solvers = {}
    for name in solver_names:
        _worker_solvers[name] = SOLVERS[name]()


def _process_puzzle(puzzle_data):
    """Worker function to process a single puzzle with all solvers.

    Returns (result_dict, timing_dict).
    """
    (puzzle_dict, processed_clues, processed_grid, solver_names, sampling_ratio_runs, max_steps) = (
        puzzle_data
    )

    result_dict = dict(puzzle_dict)
    result_dict["traces"] = {}
    timing = {}

    for solver_name in solver_names:
        solver = _worker_solvers[solver_name]
        solver_dict = {}
        t0 = time.time()

        for sampling_ratio, number_of_samples in sampling_ratio_runs:
            runs = []
            for _ in range(number_of_samples):
                steps, steps_to_solve, cr_losses, mse_losses, num_steps, solved, step_ratio = (
                    solver.try_solve(
                        clues=processed_clues,
                        grid=processed_grid,
                        sampling_ratio=sampling_ratio,
                        max_steps=max_steps,
                    )
                )
                runs.append(
                    {
                        "steps": [step for step in steps],
                        "steps_to_solve": steps_to_solve,
                        "cr_losses": cr_losses,
                        "mse_losses": mse_losses,
                        "num_steps": num_steps,
                        "solved": solved,
                        "step_ratio": step_ratio,
                    }
                )
            solver_dict[sampling_ratio] = runs
        result_dict["traces"][solver_name] = solver_dict
        timing[solver_name] = time.time() - t0

    return result_dict, timing


def _generate_puzzle_args(dataset, solver_names, sampling_ratio_runs, max_steps):
    """Generator that yields puzzle data for parallel processing."""
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, collate_fn=collate_combined
    )

    for X, y, _, X_raw, y_raw, meta in dataloader:
        processed_clues = X.squeeze(0).reshape(2, 5, 3)
        processed_grid = y.squeeze(0)

        puzzle_dict = {
            "clues": X_raw[0],
            "grid": y_raw[0],
            "meta": {
                "shape": meta[0]["shape"],
                "density": meta[0]["density"],
                "mean_clue_runs": meta[0]["mean_clue_runs"],
            },
        }

        yield (
            puzzle_dict,
            processed_clues,
            processed_grid,
            solver_names,
            sampling_ratio_runs,
            max_steps,
        )


def _flush_batch(f, batch_records):
    """Write a batch of records to the output file."""
    for record in batch_records:
        f.write(json.dumps(to_serialisable(record)) + "\n")
    f.flush()


def _print_timing_table(all_timings, total_puzzles):
    """Print a summary table of per-solver timing."""
    if not all_timings:
        return
    # Aggregate
    totals = defaultdict(float)
    for t in all_timings:
        for name, elapsed in t.items():
            totals[name] += elapsed

    print(f"\n{'Solver':<35} {'Avg (s)':>10} {'Total (s)':>10} {'% Total':>8}")
    print("-" * 65)
    grand_total = sum(totals.values())
    for name in sorted(totals.keys(), key=lambda n: -totals[n]):
        avg = totals[name] / total_puzzles
        pct = 100 * totals[name] / grand_total if grand_total > 0 else 0
        print(f"{name:<35} {avg:>10.4f} {totals[name]:>10.2f} {pct:>7.1f}%")
    print("-" * 65)
    print(f"{'TOTAL':<35} {grand_total / total_puzzles:>10.4f} {grand_total:>10.2f}")


def main(
    data_dir: str | None = None,
    input_path: str = "raw/nonogram_5x5.jsonl",
    output_path: str | None = None,
    max_size: int = 5,
    solver_names: list[str] | None = None,
    sampling_ratio_runs: list[tuple] | None = None,
    max_steps: int = 20,
    n_workers: int | None = None,
    batch_size: int = 10,
    chunksize: int | None = None,
):
    """
    Generate solving traces dataset with parallel processing.

    Args:
        data_dir: Data directory path (default: $DATA_DIR or ./data)
        input_path: Input dataset path relative to data_dir
        output_path: Output JSONL path (default: data/traces/test.jsonl)
        max_size: Maximum number of puzzles to process
        solver_names: List of solver names to use (default: all SOLVERS)
        sampling_ratio_runs: List of (sampling_ratio, num_samples) tuples
        max_steps: Maximum steps per solver run
        n_workers: Number of parallel workers (default: cpu_count)
        batch_size: Number of puzzles to batch before writing
        chunksize: Chunk size for multiprocessing (None = auto)
    """
    start_total = time.time()

    if data_dir is None:
        data_dir = os.environ.get("DATA_DIR", os.path.join(os.getcwd(), "data"))

    if output_path is None:
        output_path = os.path.join(data_dir, "traces/test.jsonl")

    if solver_names is None:
        solver_names = list(SOLVERS.keys())

    if sampling_ratio_runs is None:
        sampling_ratio_runs = SAMPLING_RATIO_RUNS

    if n_workers is None:
        n_workers = os.cpu_count() or 1
    n_workers = max(1, min(n_workers, max_size))

    if chunksize is None:
        chunksize = max(1, min(max_size // n_workers, 10))

    # Load dataset
    dataset = NonogramDataset(input_path, max_size=max_size)
    total_puzzles = len(dataset)
    print(f"Loaded {total_puzzles} puzzles")
    print(f"Solvers: {solver_names}")

    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Generate puzzle arguments
    puzzle_args_gen = _generate_puzzle_args(dataset, solver_names, sampling_ratio_runs, max_steps)

    # Process puzzles in parallel
    generated = 0
    batch_records = []
    all_timings = []

    with open(output_path, "w") as f:
        if n_workers == 1:
            # Single process mode — create solvers once, reuse
            _pool_init(solver_names)

            for puzzle_args in puzzle_args_gen:
                result, timing = _process_puzzle(puzzle_args)
                batch_records.append(result)
                all_timings.append(timing)
                generated += 1

                if len(batch_records) >= batch_size:
                    _flush_batch(f, batch_records)
                    batch_records.clear()
                    gc.collect()
                    print(f"  [{generated}/{total_puzzles}] puzzles done...")
        else:
            # Parallel processing mode
            ctx = mp.get_context()
            with ctx.Pool(
                processes=n_workers,
                initializer=_pool_init,
                initargs=(solver_names,),
            ) as pool:
                result_iter = pool.imap_unordered(
                    _process_puzzle, puzzle_args_gen, chunksize=chunksize
                )

                for result, timing in result_iter:
                    batch_records.append(result)
                    all_timings.append(timing)
                    generated += 1

                    if len(batch_records) >= batch_size:
                        _flush_batch(f, batch_records)
                        batch_records.clear()
                        gc.collect()
                        if generated % 50 == 0 or generated == total_puzzles:
                            print(f"  [{generated}/{total_puzzles}] puzzles done...")

        # Final flush inside the with block
        if batch_records:
            _flush_batch(f, batch_records)
            batch_records.clear()

    end_total = time.time()
    elapsed = end_total - start_total

    print("\n=== Generation completed ===")
    print(f"Workers: {n_workers}  chunksize: {chunksize}")
    print(f"Puzzles: {generated}")
    print(f"Time:    {elapsed:.2f}s")

    _print_timing_table(all_timings, generated)
    print(f"\nOutput: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate solving traces dataset with parallel processing"
    )
    parser.add_argument(
        "-d",
        "--data-dir",
        type=str,
        default=None,
        help="Data directory path (default: $DATA_DIR or ./data)",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="raw/nonogram_5x5.jsonl",
        help="Input dataset path (default: raw/nonogram_5x5.jsonl)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output JSONL path (default: data/traces/test.jsonl)",
    )
    parser.add_argument(
        "-n",
        "--max-size",
        type=int,
        default=5,
        help="Maximum number of puzzles to process (default: 5)",
    )
    parser.add_argument(
        "--solvers", nargs="+", default=None, help="Solver names to use (default: all)"
    )
    parser.add_argument(
        "--max-steps", type=int, default=20, help="Maximum steps per solver run (default: 20)"
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: cpu_count)",
    )
    parser.add_argument(
        "-b", "--batch-size", type=int, default=10, help="Batch size for writing (default: 10)"
    )
    parser.add_argument(
        "-c",
        "--chunksize",
        type=int,
        default=None,
        help="Chunk size for multiprocessing (default: auto)",
    )

    args = parser.parse_args()

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    main(
        data_dir=args.data_dir,
        input_path=args.input,
        output_path=args.output,
        max_size=args.max_size,
        solver_names=args.solvers,
        max_steps=args.max_steps,
        n_workers=args.workers,
        batch_size=args.batch_size,
        chunksize=args.chunksize,
    )
