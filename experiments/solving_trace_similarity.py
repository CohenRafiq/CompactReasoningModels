import torch
import numpy as np
import itertools
from collections import defaultdict

from compactreasoningmodels.solving_traces.random import RandomSolver
from compactreasoningmodels.solving_traces.cheating_random import CheatingRandom
from compactreasoningmodels.datasets.jsonl import JsonlDataset

class BucketDictionary:
    def __init__(self, n=10):
        self.n, self._b = n, defaultdict(list)
    
    def add(self, key, val):
        self._b[min(int(key * self.n), self.n - 1)].append(val)
    
    def get(self, key):
        return self._b.get(min(int(key * self.n), self.n - 1), [])

def main(number_samples=1000):
    dataset = JsonlDataset("raw/nonogram_5x5_small.jsonl", target_shape=(5, 5))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False) 
    solver_list = {"random": RandomSolver, "cheating_random": CheatingRandom}
    heatmap_methods = ["random", "cheating_random"]
    sample_methods = ["random", "cheating_random"]

    heatmap_steps = {name: [] for name in heatmap_methods}
    discrete_steps = {name: [] for name in sample_methods}
    for i, (input_tensor, target_tensor) in enumerate(dataloader):
        if i >= number_samples:
            break
        clues = input_tensor.squeeze(0).numpy()
        target = target_tensor.squeeze(0).numpy()

        initial_grid = np.full(target.shape, 0.5, dtype=float)
        initialised_solvers = {name: solver_cls(clues, target.shape, initial_grid) for name, solver_cls in solver_list.items()}
        initialised_solvers["cheating_random"].set_correct(target)

        for name, solver in initialised_solvers.items():
            heatmap_steps[name].append(solver.heatmap_step(initial_grid))
            discrete_steps[name].append(solver.step(initial_grid))

    difference_buckets = {(heatmap, sample): BucketDictionary(n=10) for heatmap, sample in itertools.product(heatmap_methods, sample_methods)}
    for heatmap_name, heatmap_steps in heatmap_steps.items():
        for sample_name, discrete_steps_list in discrete_steps.items():
            for heatmap_step, discrete_step in zip(heatmap_steps, discrete_steps_list):
                for cell_index in range(np.prod(target.shape)):
                    heatmap_value = np.array([step.flatten()[cell_index] for step in heatmap_step])
                    discrete_value = np.array([step.flatten()[cell_index] for step in discrete_step])
                    difference_buckets[(heatmap_name, sample_name)].add(heatmap_value, discrete_value)

    for (heatmap_name, sample_name), bucket_dict in difference_buckets.items():
        print(f"Heatmap: {heatmap_name}, Sample: {sample_name}")
        for bucket_index in range(bucket_dict.n):
            values = bucket_dict.get(bucket_index / bucket_dict.n)
            if values:
                mean_difference = np.mean(values)
                print(f"  Bucket {bucket_index}: Mean Difference = {mean_difference:.4f}, Count = {len(values)}")
            else:
                print(f"  Bucket {bucket_index}: No data")



if __name__ == "__main__":
    main()