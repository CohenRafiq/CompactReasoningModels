import os
import sys
import time
import itertools
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

def solve(row_clues, col_clues, constraint_propagator, known_grid, prev, rng):
    grid = constraint_propagator.solve_grid(row_clues, col_clues, known_grid)
    if grid is None:
        return None

    determined = (grid == 0) | (grid == 1)
    if determined.all():
        return [(grid, "cprop")]

    if np.array_equal(grid, prev):
        mask = ~determined
        coords = np.argwhere(mask)
        vals = grid[mask]
        dist = np.abs(vals - 0.5)
        tied = np.flatnonzero(dist == dist.max())
        pick = tied[rng.integers(len(tied))] if len(tied) > 1 else tied[0]
        r, c = coords[pick]

        test_1_grid = np.copy(known_grid)
        test_1_grid[r, c] = 1
        solution_1 = solve(row_clues, col_clues, constraint_propagator, test_1_grid, grid, rng)

        test_0_grid = np.copy(known_grid)
        test_0_grid[r, c] = 0
        solution_0 = solve(row_clues, col_clues, constraint_propagator, test_0_grid, grid, rng)

        if solution_1 is not None and solution_0 is None:
            return solution_1 + [(grid, "search")]
        elif solution_1 is None and solution_0 is not None:
            return solution_0 + [(grid, "search")]
        else:
            return None
    else:
        new_known_grid = np.where(determined, grid, -1).astype(np.int64)
        solution = solve(row_clues, col_clues, constraint_propagator, new_known_grid, grid, rng)
        if solution is None:
            return None
        return solution + [(grid, "cprop")]


def generate_puzzle(clue_generator, constraint_propagator, rng, max_attempts=10000):
    row_clues, col_clues = clue_generator.gen_clues(max_attempts=max_attempts)
    grid_density = sum(x for inner in row_clues for x in inner) / (clue_generator.h * clue_generator.w)
    known_grid = np.full((clue_generator.h, clue_generator.w), -1, dtype=np.int64)
    prev = np.full((clue_generator.h, clue_generator.w), -1, dtype=np.int64)

    solutions = solve(row_clues, col_clues, constraint_propagator, known_grid, prev, rng)
    if not solutions:
        return None

    solutions = list(reversed(solutions))
    intermediate_grids = [g.tolist() for g, _ in solutions]
    intermediate_methods = [m for _, m in solutions]

    requires_search = any(method == "search" for _, method in solutions)
    one_step_rounding = _check_one_step_rounding(intermediate_grids)

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
        "one_step_rounding": one_step_rounding,
    }


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


def _pool_init(width, height, density, master_seed):
    global _worker_clue_gen, _worker_prop, _worker_rng
    seed_seq = np.random.SeedSequence([master_seed, os.getpid()])
    _worker_rng = np.random.default_rng(seed_seq)
    np.random.seed(seed_seq.generate_state(1)[0])
    _worker_clue_gen = ClueGenerator(width, height, density)
    _worker_prop = ConstraintPropagator()


def _generate_one(max_attempts, safety_cap=100_000):
    attempts = 0
    while True:
        attempts += 1
        puzzle = generate_puzzle(_worker_clue_gen, _worker_prop, _worker_rng, max_attempts=max_attempts)
        if puzzle is not None:
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


def main(num_samples: int = 10_000,
         batch_size: int = 1_000,
         parquet_path: str = "puzzles.parquet",
         width: int = 5,
         height: int = 5,
         density: float = 0.5,
         n_workers: int | None = None,
         chunksize: int | None = None,  # Now a FIXED value, not auto-scaled
         master_seed: int = 42,
         max_attempts: int = 10_000):
    
    start_total = time.time()

    if n_workers is None:
        n_workers = os.cpu_count() or 1
    n_workers = max(1, min(n_workers, num_samples))

    # CRITICAL FIX: Fixed chunksize, NOT scaled with num_samples
    # For variable-duration tasks (puzzle solving), small chunks prevent starvation
    if chunksize is None:
        chunksize = 10  # Fixed small number - tune based on your task variance
    else:
        chunksize = max(1, chunksize)

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
        ("one_step_rounding", pa.bool_())
    ])

    writer = pq.ParquetWriter(parquet_path, schema, compression="zstd")
    batch_records = []
    generated = 0
    attempted = 0

    if n_workers == 1:
        _pool_init(width, height, density, master_seed)
        result_stream = (_generate_one(max_attempts) for _ in range(num_samples))
        pool_ctx = None
    else:
        pool_ctx = mp.Pool(processes=n_workers, initializer=_pool_init,
                            initargs=(width, height, density, master_seed))
        
        # CRITICAL FIX: Use imap_unordered with fixed small chunksize
        # imap_unordered lets workers return results as they finish, 
        # preventing the "slow chunk blocks everything" problem
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
                "one_step_rounding": puzzle["one_step_rounding"],
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
    main(num_samples=1_000, batch_size=100, width=n, height=n, density=0.5, 
         chunksize=100, parquet_path="data/raw/nng_test.parquet")