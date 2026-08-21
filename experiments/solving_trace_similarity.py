import torch
import numpy as np
import itertools

from compactreasoningmodels.solving_traces.random_solver import RandomSolver
from compactreasoningmodels.solving_traces.cheating_random import CheatingRandom
from compactreasoningmodels.datasets.jsonl import JsonlDataset


def changed_cells_metric(initial_grids, heatmaps, samples):
    """
    For each puzzle, find cells where sample differs from initial.
    Check if heatmap predicted that change correctly.
    """
    n = len(initial_grids)
    correct_predictions = 0
    total_changes = 0
    
    # Also track: did heatmap predict ANY change at all?
    heatmap_predicted_change = 0
    
    for init, hm, sample in zip(initial_grids, heatmaps, samples):
        # Find changed cells
        changed = (sample != init)
        n_changed = changed.sum()
        total_changes += n_changed
        
        # For changed cells: did heatmap predict the new value?
        for idx in np.argwhere(changed):
            i, j = idx
            # Heatmap's expected value for this cell
            predicted = hm[i, j]
            actual = sample[i, j]
            initial = init[i, j]
            
            # Did heatmap predict a change in the right direction?
            # If initial=0.3, sample=1, did hm predict > 0.3?
            if actual > initial and predicted > initial:
                correct_predictions += 1
            elif actual < initial and predicted < initial:
                correct_predictions += 1
            
            # Did heatmap predict any change at all?
            if abs(predicted - initial) > 0.01:
                heatmap_predicted_change += 1
    
    change_accuracy = correct_predictions / total_changes if total_changes > 0 else 0
    change_detection_rate = heatmap_predicted_change / total_changes if total_changes > 0 else 0
    
    return {
        'total_changes': total_changes,
        'correct_direction': correct_predictions,
        'change_accuracy': change_accuracy,
        'change_detection_rate': change_detection_rate,
    }


def top_k_accuracy(initial_grids, heatmaps, samples, k=1):
    """
    For each puzzle, find the cell with largest predicted change in heatmap.
    Check if that cell was actually changed in the sample.
    """
    n = len(initial_grids)
    correct = 0
    
    for init, hm, sample in zip(initial_grids, heatmaps, samples):
        # Predicted change magnitude
        predicted_change = np.abs(hm - init)
        
        # Top-k cells by predicted change
        flat_idx = np.argsort(predicted_change.flatten())[-k:]
        top_k_cells = [(idx // hm.shape[1], idx % hm.shape[1]) for idx in flat_idx]
        
        # Actual changed cells
        actual_changed = set(map(tuple, np.argwhere(sample != init)))
        
        # Did any top-k cell actually change?
        if any(cell in actual_changed for cell in top_k_cells):
            correct += 1
    
    return correct / n


def change_mse(initial_grids, heatmaps, samples):
    """
    MSE computed only over cells that actually changed.
    """
    mses = []
    for init, hm, sample in zip(initial_grids, heatmaps, samples):
        changed = (sample != init)
        if changed.sum() == 0:
            continue
        mse = np.mean((hm[changed] - sample[changed]) ** 2)
        mses.append(mse)
    return np.mean(mses) if mses else 0


def change_correlation(initial_grids, heatmaps, samples):
    """
    Correlation between predicted change (hm - init) and actual change (sample - init).
    """
    pred_changes = []
    actual_changes = []
    
    for init, hm, sample in zip(initial_grids, heatmaps, samples):
        pred_changes.extend((hm - init).flatten())
        actual_changes.extend((sample - init).flatten())
    
    pred_changes = np.array(pred_changes)
    actual_changes = np.array(actual_changes)
    
    return np.corrcoef(pred_changes, actual_changes)[0, 1]


def main(number_samples=100):
    dataset = JsonlDataset("raw/nonogram_5x5_small.jsonl", target_shape=(5, 5))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    solver_list = {"random": RandomSolver, "cheating_random": CheatingRandom}
    methods = ["random", "cheating_random"]

    # Store: initial_grid, heatmap, sample for each puzzle and each solver
    puzzle_data = []
    
    for i, (input_tensor, target_tensor) in enumerate(dataloader):
        if i >= number_samples:
            break
        clues = input_tensor.squeeze(0).numpy()
        target = target_tensor.squeeze(0).numpy()
        initial_grid = np.random.uniform(0.1, 0.9, size=target.shape)
        
        solvers = {name: solver_cls(clues, target.shape, initial_grid)
                   for name, solver_cls in solver_list.items()}
        solvers["cheating_random"].set_correct(target)
        
        puzzle_result = {'initial': initial_grid}
        for name, solver in solvers.items():
            hm = solver.heatmap_step(initial_grid)
            sample = solver.step(initial_grid)
            puzzle_result[name] = (hm, sample)
        
        puzzle_data.append(puzzle_result)

    print("=" * 80)
    print("Changed-Cell Metrics (focusing on cells that actually changed)")
    print("=" * 80)
    
    for h_name in methods:
        for s_name in methods:
            inits = [p['initial'] for p in puzzle_data]
            hms = [p[h_name][0] for p in puzzle_data]
            samples = [p[s_name][1] for p in puzzle_data]
            
            metrics = changed_cells_metric(inits, hms, samples)
            top1 = top_k_accuracy(inits, hms, samples, k=1)
            top3 = top_k_accuracy(inits, hms, samples, k=3)
            cmse = change_mse(inits, hms, samples)
            corr = change_correlation(inits, hms, samples)
            
            marker = " *** MATCH" if h_name == s_name else ""
            print(f"\n{h_name:15s} → {s_name:15s}{marker}")
            print(f"  Total changes:          {metrics['total_changes']}")
            print(f"  Correct direction:      {metrics['correct_direction']}")
            print(f"  Change accuracy:        {metrics['change_accuracy']:.4f}")
            print(f"  Change detection rate:  {metrics['change_detection_rate']:.4f}")
            print(f"  Top-1 accuracy:         {top1:.4f}")
            print(f"  Top-3 accuracy:         {top3:.4f}")
            print(f"  Change MSE:             {cmse:.6f}")
            print(f"  Change correlation:     {corr:.6f}")

    print("\n" + "=" * 80)
    print("Expected Results:")
    print("  random → random:       low change accuracy (~0.5), low top-k")
    print("                         (heatmap predicts uniform tiny nudge)")
    print("  cheat → cheat:         HIGH change accuracy (~1.0), HIGH top-k")
    print("                         (heatmap exactly predicts which wrong cell)")
    print("  cross:                 intermediate (some correlation by chance)")
    print("=" * 80)


if __name__ == "__main__":
    main()