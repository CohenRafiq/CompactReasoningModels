import pyarrow.parquet as pq
import pathlib

def parquet_row_count(path: str) -> int:
    """
    Returns the total number of rows in a Parquet dataset.
    Handles a single file or a directory of Parquet files.
    """
    p = pathlib.Path(path)

    if p.is_file():                     # single .parquet file
        pf = pq.ParquetFile(str(p))
        return pf.metadata.num_rows

    # directory – sum rows from each file
    total = 0
    for file in p.rglob("*.parquet"):
        pf = pq.ParquetFile(str(file))
        total += pf.metadata.num_rows
    return total

# Example
print(parquet_row_count("data/raw/nng_15x15_large.parquet"))

# train_loader, test_loader = pq_dataset.create_dataloaders(batch_size=32)
# print(f"Train loader size: {len(train_loader.dataset)}")
# print(f"Test loader size: {len(test_loader.dataset)}")

# i = 0
# for data in pq_dataset:
#     if data["requires_search"]:
#         print(f"Puzzle {i + 1}:")
#         print("Row clues:\n", data["row_clues"], "\n")
#         print("Column clues:\n", data["col_clues"], "\n")
#         # print("Solution:", data["solution"])
#         print("=" * 20)
#         i += 1
#         if i >= 5:  # Print only the first 5 examples
#             break
    

import pyarrow.parquet as pq
import pyarrow as pa

# Load the table (or use an existing one)
table = pq.read_table("data/raw/nng_15x15_large.parquet", columns=["requires_search"])

# Convert the column to a NumPy array of bools
requires_search = table.column("requires_search").to_numpy()

true_count = requires_search.sum()
total_rows = len(requires_search)

percent_true = (true_count / total_rows) * 100 if total_rows else 0.0
print(f"requires_search is True for {percent_true:.2f}% of the instances "
      f"({true_count:,} / {total_rows:,} rows).")