import time
import json
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from clue_generator import ClueGenerator
from constraint_propagator import ConstraintPropagator

rng = np.random.default_rng(42)

def solve(row_clues, col_clues, constraint_propagator, known_grid, prev):
    grid = constraint_propagator.solve_grid(row_clues, col_clues, known_grid)
    if grid is None:
        return None
    if np.isin(grid, [0, 1]).all():
        return [(grid, "cprop")]
    if np.array_equal(grid, prev):
        mask = (grid != 0) & (grid != 1)
        coords = np.argwhere(mask)
        vals = grid[mask]
        shuffled_idx = rng.permutation(len(vals))
        shuffled_vals = vals[shuffled_idx]
        shuffled_coords = coords[shuffled_idx]
        dist = np.abs(shuffled_vals - 0.5)
        sort_idx = np.argsort(-dist, kind='stable')
        sorted_coords = shuffled_coords[sort_idx]

        for r, c in sorted_coords:
            test_1_grid = np.copy(known_grid)
            test_1_grid[r, c] = 1
            solution_1 = solve(row_clues, col_clues, constraint_propagator, test_1_grid, grid)

            test_0_grid = np.copy(known_grid)
            test_0_grid[r, c] = 0
            solution_0 = solve(row_clues, col_clues, constraint_propagator, test_0_grid, grid)

            if solution_1 is None and solution_0 is None:
                return None
            elif solution_1 is not None and solution_0 is None:
                return solution_1 + [(grid, "search")]
            elif solution_1 is None and solution_0 is not None:
                return solution_0 + [(grid, "search")]
            else:
                return None # The puzzle has multiple solutions, so we cannot solve it uniquely.
    else:
        new_known_grid = np.full_like(grid, -1, dtype=np.int64)
        new_known_grid[grid == 0] = 0
        new_known_grid[grid == 1] = 1
        solution = solve(row_clues, col_clues, constraint_propagator, new_known_grid, grid)
        if solution is None:
            return None
        else:
            return solution + [(grid, "cprop")]

def generate_puzzle(clue_generator: ClueGenerator,
                    constraint_propagator: ConstraintPropagator,
                    max_attempts: int = 10000):
    clues = clue_generator.gen_clues(max_attempts=max_attempts)
    row_clues, col_clues = clues
    grid_density = sum(x for inner in row_clues for x in inner) / (clue_generator.h * clue_generator.w)
    known_grid = np.full((clue_generator.h, clue_generator.w), -1, dtype=np.int64)
    prev = np.full((clue_generator.h, clue_generator.w), -1, dtype=np.int64)
    solutions = solve(row_clues, col_clues, constraint_propagator, known_grid, prev)
    requires_search = any(method == "search" for _, method in solutions) if solutions else False
    if not solutions:
        return None
    return {
        "row_clues": row_clues,
        "col_clues": col_clues,
        "solution": solutions[0][0],
        "intermediate_solutions": solutions,
        "grid_density": float(grid_density),
        "grid_height": clue_generator.h,
        "grid_width": clue_generator.w,
        "steps": len(solutions),
        "requires_search": requires_search
    }

def array_to_json(arr):
    """Convert a NumPy array (2‑D) to a nested Python list and then to JSON."""
    return json.dumps(arr.tolist())

def list_to_json(lst):
    """Serialise any Python list (including nested) to JSON."""
    return json.dumps(lst)

def main(num_samples: int = 10_000,
         batch_size: int = 1_000,
         parquet_path: str = "puzzles.parquet",
         width: int = 5,
         height: int = 5,
         density: float = 0.5):
    start_total = time.time()

    # Initialise the generators once – you can customise the dimensions here.
    clue_gen = ClueGenerator(width, height, density)          # <-- adjust constructor args if needed
    prop = ConstraintPropagator()       # <-- adjust constructor args if needed

    # Define the Arrow schema (all columns are stored as primitive types or JSON strings)
    schema = pa.schema([
        ("puzzle_id", pa.int64()),
        ("row_clues", pa.string()),          # JSON
        ("col_clues", pa.string()),          # JSON
        ("solution", pa.string()),           # JSON (nested list of ints)
        ("intermediate_solutions", pa.string()),  # JSON (list of [grid, method] pairs)
        ("grid_density", pa.float64()),
        ("grid_height", pa.int32()),
        ("grid_width", pa.int32()),
        ("steps", pa.int32()),
        ("requires_search", pa.bool_())
    ])

    writer = pq.ParquetWriter(parquet_path, schema, compression="zstd")
    batch_records = []
    generated = 0
    attempted = 0

    while generated < num_samples:
        attempted += 1
        puzzle = generate_puzzle(clue_gen, prop)
        if puzzle is None:
            # Skip failed attempts – you may want to log or count them.
            continue

        # Serialise complex fields as JSON strings.
        record = {
            "puzzle_id": generated,
            "row_clues": list_to_json(puzzle["row_clues"]),
            "col_clues": list_to_json(puzzle["col_clues"]),
            "solution": array_to_json(puzzle["solution"]),
            "intermediate_solutions": list_to_json(
                [(array_to_json(g), m) for g, m in puzzle["intermediate_solutions"]]
            ),
            "grid_density": puzzle["grid_density"],
            "grid_height": puzzle["grid_height"],
            "grid_width": puzzle["grid_width"],
            "steps": puzzle["steps"],
            "requires_search": puzzle["requires_search"]
        }
        batch_records.append(record)
        generated += 1

        # When we have a full batch, write it to Parquet.
        if len(batch_records) >= batch_size:
            table = pa.Table.from_pydict({k: [r[k] for r in batch_records] for k in record}, schema=schema)
            writer.write_table(table)
            batch_records.clear()
            print(f"✔️  Written {generated}/{num_samples} puzzles...")

    # Write any remaining records that didn't fill a full batch.
    if batch_records:
        table = pa.Table.from_pydict({k: [r[k] for r in batch_records] for k in record}, schema=schema)
        writer.write_table(table)
        batch_records.clear()

    writer.close()
    end_total = time.time()
    elapsed = end_total - start_total
    print("\n=== Generation & Parquet write completed ===")
    print(f"Total puzzles generated: {generated}")
    print(f"Total attempts (including failures): {attempted}")
    print(f"Time elapsed: {elapsed:.2f} seconds")
    print(f"Parquet file written to: {parquet_path}")

if __name__ == "__main__":
    # Adjust the arguments if you want a different sample count or batch size.
    main(num_samples=1_000, batch_size=100, width=15, height=15, density=0.5, parquet_path="data/raw/nng_15x15.parquet")