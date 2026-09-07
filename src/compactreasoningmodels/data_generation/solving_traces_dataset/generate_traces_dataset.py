import os
import torch
import json
import numpy as np
import time

from compactreasoningmodels.solvers import SOLVERS
from compactreasoningmodels.datasets.nonogram_dataset import NonogramDataset
from compactreasoningmodels.datasets.collate import collate_combined

TEMP_SOLVERS = {
    # "mac": SOLVERS["mac"],
    # "genetic_algorithm_det": SOLVERS["genetic_algorithm_det"],
    "gradient_descent_global_adam": SOLVERS["gradient_descent_global_adam"],
    "gradient_descent_global_sgd": SOLVERS["gradient_descent_global_sgd"],
    # "model_solver": SOLVERS["model_solver"],
    # "genetic_algorithm_dep": SOLVERS["genetic_algorithm_dep"],
}

SAMPLING_RATIO_RUNS = [(1.0, 1), (0.8, 3), (0.6, 3), (0.4, 3)]

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

def main():
    os.environ["DATA_DIR"] = os.path.join(os.getcwd(), "data")
    dataset = NonogramDataset("raw/nonogram_5x5.jsonl", max_size=50)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_combined)
    output_path = os.path.join(os.environ["DATA_DIR"], "traces/test.jsonl")
    
    with open(output_path, "w") as f:
        for i, (X, y, padding_mask, X_raw, y_raw, meta) in enumerate(dataloader):
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
                "traces": {}
            }

            for solver_name, solver_class in SOLVERS.items():
                start = time.time()
                solver = solver_class()
                solver_dict = {}

                for sampling_ratio, number_of_samples in SAMPLING_RATIO_RUNS:
                    runs = []
                    for _ in range(number_of_samples):
                        steps, steps_to_solve, cr_losses, mse_losses, num_steps, solved, step_ratio = solver.try_solve(
                            clues=processed_clues,
                            grid=processed_grid,
                            sampling_ratio=sampling_ratio,
                            max_steps=20
                        )
                        runs.append({
                            "steps": [step for step in steps],
                            "steps_to_solve": steps_to_solve,
                            "cr_losses": cr_losses,
                            "mse_losses": mse_losses,
                            "num_steps": num_steps,
                            "solved": solved,
                            "step_ratio": step_ratio,
                        })
                    solver_dict[sampling_ratio] = runs
                puzzle_dict["traces"][solver_name] = solver_dict
                print(" Solver:", solver_name, "Time taken:", time.time() - start)
            f.write(json.dumps(to_serialisable(puzzle_dict)) + "\n")
            print(f"Processed puzzle {i+1}/{len(dataloader)}")
                
          

if __name__ == "__main__":
    main()