import numpy as np
import pandas as pd
import torch

from pipeline.trace_comparison.src.trace_comparison import (
    CKABlock,
    DTWBlock,
    FlattenBlock,
    MDSBlock,
    MSEBlock,
)
from pipeline.trace_comparison.src.trace_comparison.experiments.solver_similarity import (  # type: ignore[attr-defined]
    FullDatasetSimilarityExperiment as Experiment,
)


def generate_pair(n, k):
    a = np.random.randint(0, n, size=k)
    b = np.random.randint(0, n, size=k)
    while np.any(b == a):
        collide = b == a
        b[collide] = np.random.randint(0, n, size=collide.sum())
    return a, b


def extract_steps(traces_dict):
    if not isinstance(traces_dict, dict):
        return None
    result = {}
    for solver, versions in traces_dict.items():
        try:
            result[solver] = versions["1.0"][0]["steps"]
        except (KeyError, IndexError, TypeError):
            result[solver] = None
    return result


def load_data(location, k=200, seed=None):
    if seed is not None:
        np.random.seed(seed)

    df = pd.read_json(location, lines=True)
    df = df["traces"].apply(extract_steps).apply(pd.Series)
    idx_left, idx_right = generate_pair(len(df), k)

    output = {}
    for solver in df.columns:
        valid = df[solver].dropna()
        if len(valid) < max(idx_left.max(), idx_right.max()) + 1:
            raise ValueError(f"Solver {solver} has insufficient data")

        left = valid.iloc[idx_left].values
        right = valid.iloc[idx_right].values
        output[solver] = (left, right)
    return output


def _extract_values(result):
    """Normalize an Experiment result into a flat numpy array of per-pair scores."""
    if torch.is_tensor(result):
        arr = result.detach().cpu().numpy()
    elif isinstance(result, (tuple, list)):
        arr = np.asarray(result)
    else:
        arr = np.asarray(result)
    return arr.reshape(-1)


def build_blocks(experiment):
    match experiment:
        case "CKA":
            return [FlattenBlock(start_dim=2), CKABlock()]
        case "MSE":
            return [MSEBlock()]
        case "DTW":
            return [FlattenBlock(start_dim=2), DTWBlock(), MDSBlock(n_components=2), CKABlock()]
        case _:
            raise ValueError(f"Unknown experiment: {experiment}")


def compute_similarity_grids(data, experiment="CKA"):
    solvers = list(data.keys())

    mean_grid = pd.DataFrame(index=solvers, columns=solvers, dtype=float)
    std_grid = pd.DataFrame(index=solvers, columns=solvers, dtype=float)

    for i, solver1 in enumerate(solvers):
        for solver2 in solvers[i:]:
            left = data[solver1][0]
            right = data[solver2][1]

            left_arr = [np.asarray(item, dtype=np.float32) for item in left]
            right_arr = [np.asarray(item, dtype=np.float32) for item in right]
            torch_data = torch.tensor(np.stack([left_arr, right_arr], axis=0), dtype=torch.float32)

            # Fresh Experiment instance per pair to avoid state leaking across runs
            exp = Experiment(blocks=build_blocks(experiment), name=f"{solver1} vs {solver2}")
            result = exp.run(torch_data)
            values = _extract_values(result)

            mean_val = float(np.mean(values))
            std_val = float(np.std(values))

            mean_grid.loc[solver1, solver2] = mean_val
            std_grid.loc[solver1, solver2] = std_val

            if solver1 != solver2:
                mean_grid.loc[solver2, solver1] = mean_val
                std_grid.loc[solver2, solver1] = std_val

    return {"mean": mean_grid, "std": std_grid}


def main(experiment="DTW"):
    data = load_data("data/traces/large_dataset.jsonl")
    grids = compute_similarity_grids(data, experiment=experiment)
    return grids


if __name__ == "__main__":
    grids = main()
    print(grids["mean"])
