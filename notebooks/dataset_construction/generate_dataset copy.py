import os
import time
import multiprocessing as mp
import gc
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from clue_generator import ClueGenerator
from constraint_propagator import ConstraintPropagator

try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

class MultipleSoltionsFound(Exception):
    """Raised to abort recursion as soon as a solution is discovered."""
    pass

def solve(row_clues, col_clues, constraint_propagator, known_grid, true_grid, counterexample=False,
          prev_combined=None, prev_known_grid=None,
          prev_row_grid=None, prev_col_grid=None, rng=None):
    """
    Recursive solver with incremental constraint propagation.
    
    Only rows / columns that changed since the previous call are recomputed.
    
    Note on variables:
    - known_grid: The strict mask of known cells. Values in {-1, 0, 1}.
    - grid: The probability grid returned by the propagator. Values in [0.0, 1.0].
    - prev_combined: The previous probability grid (used for stagnation detection).
    """

    # ------------------------------------------------------------------
    # 1. Determine which rows & columns became dirty since last step
    # ------------------------------------------------------------------
    if prev_known_grid is not None:
        changed_mask = (known_grid != prev_known_grid)
        dirty_rows = set(int(i) for i in np.where(changed_mask.any(axis=1))[0])
        dirty_cols = set(int(j) for j in np.where(changed_mask.any(axis=0))[0])
    else:
        dirty_rows = None
        dirty_cols = None

    # ------------------------------------------------------------------
    # 2. Incremental propagation
    # ------------------------------------------------------------------
    result = constraint_propagator.solve_grid(
        row_clues, col_clues, known_grid,
        return_intermediate=True,
        prev_row_grid=prev_row_grid,
        prev_col_grid=prev_col_grid,
        dirty_rows=dirty_rows,
        dirty_cols=dirty_cols,
    )
    if result[0] is None:
        return None
    
    # 'grid' contains probabilities (floats between 0 and 1)
    grid, row_grid, col_grid = result

    # ------------------------------------------------------------------
    # 3. Check for complete solution / contradiction
    # ------------------------------------------------------------------
    determined = (grid == 0) | (grid == 1)
    if determined.all():
        final_grid = grid.astype(np.int64)
        if counterexample:
            raise MultipleSoltionsFound("Multiple solutions found.")
        elif np.array_equal(final_grid, true_grid):
            # Return the float grid for consistency with intermediate steps
            return [(grid, "cprop")]
        else:
            raise ValueError("Contradiction: fully determined grid does not match true solution.")

    # ------------------------------------------------------------------
    # 4. Stagnation check – if nothing moved, we must search
    # ------------------------------------------------------------------
    if prev_combined is not None and np.array_equal(grid, prev_combined):
        mask = ~determined
        coords = np.argwhere(mask)
        vals = grid[mask]
        dist = np.abs(true_grid[mask] - vals)
        tied = np.flatnonzero(dist == dist.max())
        pick = tied[rng.integers(len(tied))] if len(tied) > 1 else tied[0]
        r, c = coords[pick]

        counterexample_known = np.copy(known_grid)
        counterexample_known[r, c] = 1 - true_grid[r, c]
        
        counterexample_solution = solve(
            row_clues, col_clues, constraint_propagator, counterexample_known, true_grid=true_grid, counterexample=True,
            prev_combined=grid.copy(), 
            prev_known_grid=known_grid.copy(),
            prev_row_grid=row_grid.copy(), 
            prev_col_grid=col_grid.copy(), 
            rng=rng
        )
        if counterexample_solution is not None:
            raise ValueError("Contradiction: counterexample solution found, but should not exist.")
            
        fixed_known = np.copy(known_grid)
        fixed_known[r, c] = true_grid[r, c]
        
        solution = solve(
            row_clues, col_clues, constraint_propagator, fixed_known, true_grid=true_grid, counterexample=counterexample,
            prev_combined=grid.copy(), 
            prev_known_grid=known_grid.copy(),
            prev_row_grid=row_grid.copy(), 
            prev_col_grid=col_grid.copy(), 
            rng=rng
        )
        if solution is None and not counterexample:
            raise ValueError("Contradiction: no solution found after fixing a cell.")
        if solution is None and counterexample:
            raise MultipleSoltionsFound("Multiple solutions found.")
        return solution + [(grid, "search")]

    else:
        # ------------------------------------------------------------------
        # 5. Constraint propagation made progress – iterate
        # ------------------------------------------------------------------
        
        # Update the known_grid mask with the newly determined cells
        new_known_grid = np.where(determined, grid, -1).astype(np.int64)
        
        solution = solve(
            row_clues, col_clues, constraint_propagator, new_known_grid,
            true_grid=true_grid, counterexample=counterexample,
            prev_combined=grid.copy(), 
            prev_known_grid=known_grid.copy(),
            prev_row_grid=row_grid.copy(), 
            prev_col_grid=col_grid.copy(), 
            rng=rng
        )
        if solution is None:
            return None
        return solution + [(grid, "cprop")]


def generate_puzzle(clue_generator, constraint_propagator, rng, max_attempts=10000):
    clues, true_grid = clue_generator.gen_clues_and_grid(max_attempts=max_attempts)
    row_clues, col_clues = clues
    grid_density = sum(x for inner in row_clues for x in inner) / (clue_generator.h * clue_generator.w)
    known_grid = np.full((clue_generator.h, clue_generator.w), -1, dtype=np.int64)
    prev = np.full((clue_generator.h, clue_generator.w), -1, dtype=np.int64)
    try:
        result = solve(
            row_clues, col_clues, constraint_propagator, known_grid, 
            true_grid=true_grid, counterexample=False,
            prev_combined=prev.copy(), prev_known_grid=None,
            prev_row_grid=None, prev_col_grid=None, rng=rng
        )
    except MultipleSoltionsFound:
        return None
    if result is None:
        raise ValueError("Failed to solve puzzle with given clues.")

    solutions = result
    solutions = list(reversed(solutions))
    intermediate_grids = [g.tolist() for g, _ in solutions]
    intermediate_methods = [m for _, m in solutions]

    requires_search = any(method == "search" for _, method in solutions)
    # one_step_rounding = _check_one_step_rounding(intermediate_grids)
    average_run_length = _average_run_length(row_clues, col_clues)

    return {
        "row_clues": row_clues,
        "col_clues": col_clues,
        "solution": solutions[-1][0].tolist(),
        "intermediate_solutions": intermediate_grids,
        "intermediate_methods": intermediate_methods,
        "grid_density": float(grid_density),
        "grid_height": clue_generator.h,
        "grid_width": clue_generator.w,
        "steps": len(solutions),
        "requires_search": requires_search,
        # "one_step_rounding": one_step_rounding,
        "average_run_length": average_run_length,
    }

def _average_run_length(row_clues, col_clues):
    """Compute the average run length of a puzzle based on its clues."""
    all_clues = row_clues + col_clues
    n = len(all_clues)
    if n == 0:
        return 0.0
    
    total = 0
    for clue in all_clues:
        # Skip first element, iterate through rest
        for cell in clue[1:]:
            if cell == 0:
                break
            total += 1
    
    return total / n + 1

def _check_one_step_rounding(intermediate_grids):
    first_grid = intermediate_grids[0]
    last_grid = intermediate_grids[-1]
    
    for i in range(len(first_grid)):
        row_first = first_grid[i]
        row_last = last_grid[i]
        for j in range(len(row_first)):
            if round(float(row_first[j])) != int(row_last[j]):
                return False
    return True


_worker_clue_gen = None
_worker_prop = None
_worker_rng = None


def _pool_init(width, height, density, master_seed, prior_cache_seed):
    global _worker_clue_gen, _worker_prop, _worker_rng
    seed_seq = np.random.SeedSequence([master_seed, os.getpid()])
    _worker_rng = np.random.default_rng(seed_seq)
    np.random.seed(seed_seq.generate_state(1)[0])
    _worker_clue_gen = ClueGenerator(width, height, density)
    _worker_prop = ConstraintPropagator(prior_cache=dict(prior_cache_seed))


def _generate_one(max_attempts, safety_cap=100_000):
    attempts = 0
    while True:
        attempts += 1
        puzzle = generate_puzzle(_worker_clue_gen, _worker_prop, _worker_rng, max_attempts=max_attempts)
        if puzzle is not None:
            _worker_prop.clear_caches()  # drops tier-2 local cache; tier-1 prior stays
            return puzzle, attempts
        if attempts >= safety_cap:
            raise RuntimeError(
                f"Exceeded {safety_cap} attempts without producing a unique-solution puzzle; "
                "check width/height/density configuration."
            )


def _generate_n(n, max_attempts):
    for _ in range(n):
        yield max_attempts


def _flush_batch(writer, batch_records, schema):
    if batch_records:
        table = pa.Table.from_pylist(batch_records, schema=schema)
        writer.write_table(table)
        batch_records.clear()
        gc.collect()


def _warm_prior_cache(width, height, density, master_seed, n_warmup=200, max_attempts=10_000):
    clue_gen = ClueGenerator(width, height, density)
    prop = ConstraintPropagator()
    rng = np.random.default_rng(np.random.SeedSequence([master_seed, 999999999]))
    for _ in range(n_warmup):
        generate_puzzle(clue_gen, prop, rng, max_attempts=max_attempts)
    return prop.prior_cache_snapshot()


def main(num_samples: int = 10_000,
         batch_size: int = 1_000,
         parquet_path: str = "puzzles.parquet",
         width: int = 5,
         height: int = 5,
         density: float = 0.5,
         n_workers: int | None = None,
         chunksize: int | None = None,
         master_seed: int = 42,
         max_attempts: int = 10_000,
         warmup_puzzles: int = 200):

    start_total = time.time()

    if n_workers is None:
        n_workers = os.cpu_count() or 1
    n_workers = max(1, min(n_workers, num_samples))

    if chunksize is None:
        chunksize = 10
    else:
        chunksize = max(1, chunksize)

    # Warm the shared tier-1 cache once, then hand it to every worker.
    print(f"Warming prior cache with {warmup_puzzles} puzzles...")
    prior_cache_seed = _warm_prior_cache(width, height, density, master_seed, n_warmup=warmup_puzzles, max_attempts=max_attempts)
    print(f"Prior cache primed with {len(prior_cache_seed)} entries.")

    schema = pa.schema([
        ("puzzle_id", pa.int64()),
        ("row_clues", pa.list_(pa.list_(pa.int16()))),
        ("col_clues", pa.list_(pa.list_(pa.int16()))),
        ("solution", pa.list_(pa.list_(pa.int8()))),
        ("intermediate_solutions", pa.list_(pa.list_(pa.list_(pa.float64())))),
        ("intermediate_methods", pa.list_(pa.string())),
        ("grid_density", pa.float64()),
        ("grid_height", pa.int32()),
        ("grid_width", pa.int32()),
        ("steps", pa.int32()),
        ("requires_search", pa.bool_()),
        # ("one_step_rounding", pa.bool_()),
        ("average_run_length", pa.float64()),
    ])

    writer = pq.ParquetWriter(parquet_path, schema, compression="zstd")
    batch_records = []
    generated = 0
    attempted = 0

    if n_workers == 1:
        _pool_init(width, height, density, master_seed, prior_cache_seed)
        result_stream = (_generate_one(max_attempts) for _ in range(num_samples))
        pool_ctx = None
    else:
        pool_ctx = mp.Pool(processes=n_workers, initializer=_pool_init,
                            initargs=(width, height, density, master_seed, prior_cache_seed))
        result_stream = pool_ctx.imap_unordered(
            _generate_one, _generate_n(num_samples, max_attempts), chunksize=chunksize
        )


    try:
        for puzzle, attempts in result_stream:
            attempted += attempts
            record = {
                "puzzle_id": generated,
                "row_clues": puzzle["row_clues"],
                "col_clues": puzzle["col_clues"],
                "solution": puzzle["solution"],
                "intermediate_solutions": puzzle["intermediate_solutions"],
                "intermediate_methods": puzzle["intermediate_methods"],
                "grid_density": puzzle["grid_density"],
                "grid_height": puzzle["grid_height"],
                "grid_width": puzzle["grid_width"],
                "steps": puzzle["steps"],
                "requires_search": puzzle["requires_search"],
                # "one_step_rounding": puzzle["one_step_rounding"],
                "average_run_length": puzzle["average_run_length"],
            }
            batch_records.append(record)
            generated += 1

            if len(batch_records) >= batch_size:
                _flush_batch(writer, batch_records, schema)
                print(f"\u2714\ufe0f  Written {generated}/{num_samples} puzzles...")
    except Exception as e:
        print(f"\n\u26a0\ufe0f  Exception after {generated} puzzles: {e}")
        _flush_batch(writer, batch_records, schema)
        raise
    finally:
        _flush_batch(writer, batch_records, schema)
        if pool_ctx is not None:
            pool_ctx.close()
            pool_ctx.join()

    writer.close()
    end_total = time.time()
    elapsed = end_total - start_total
    print("\n=== Generation & Parquet write completed ===")
    print(f"Workers used: {n_workers} (chunksize={chunksize})")
    print(f"Total puzzles generated: {generated}")
    print(f"Total attempts (including failures): {attempted}")
    print(f"Time elapsed: {elapsed:.2f} seconds")
    print(f"Parquet file written to: {parquet_path}")


if __name__ == "__main__":
    n = 15
    main(num_samples=100_000, batch_size=1_000, width=n, height=n, density=0.5, 
         chunksize=10, parquet_path="data/raw/nng_15x15_large.parquet", warmup_puzzles=1_000)