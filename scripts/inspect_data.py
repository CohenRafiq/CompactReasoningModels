import pathlib

import pyarrow.parquet as pq


def parquet_row_count(path: str) -> int:
    """
    Returns the total number of rows in a Parquet dataset.
    Handles a single file or a directory of Parquet files.
    """
    p = pathlib.Path(path)

    if p.is_file():
        pf = pq.ParquetFile(str(p))
        return pf.metadata.num_rows

    total = 0
    for file in p.rglob("*.parquet"):
        pf = pq.ParquetFile(str(file))
        total += pf.metadata.num_rows
    return total


def print_search_stats(path: str):
    table = pq.read_table(path, columns=["requires_search"])
    requires_search = table.column("requires_search").to_numpy()
    true_count = requires_search.sum()
    total_rows = len(requires_search)
    percent_true = (true_count / total_rows) * 100 if total_rows else 0.0
    print(
        f"requires_search is True for {percent_true:.2f}% of the instances "
        f"({true_count:,} / {total_rows:,} rows)."
    )


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/nng_15x15_large.parquet"
    print(f"Row count: {parquet_row_count(path)}")
    print_search_stats(path)
