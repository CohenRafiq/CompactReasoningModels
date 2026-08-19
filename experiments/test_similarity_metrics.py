from compactreasoningmodels.solving_traces.similarity_measures import *

def add_noise_to_grid(grid: list[list[int]], noise_level: float = 0.1) -> list[list[float]]:
    np.random.seed(42)
    noisy_grid = []
    for row in grid:
        noisy_row = []
        for cell in row:
            noise = np.random.uniform(-noise_level, noise_level)
            noisy_cell = min(max(cell + noise, 0), 1)  # Ensure the value stays between 0 and 1
            noisy_row.append(noisy_cell)
        noisy_grid.append(noisy_row)
    return noisy_grid

def main():
    grid1 = [[0.1, 0.9, 0.1], [0.9, 0.9, 0.1], [0.1, 0.1, 0.9]]
    grid2 = add_noise_to_grid(grid1, noise_level=0.5)
    grid3 = [[0.1, 0.9, 0.1], [0.1, 0.1, 0.1], [0.1, 0.9, 0.1]]

    measures = [MSE(), MAE(), HuberLoss(delta=1.0), MeanCosineSimilarity()]

    for measure in measures:
        print(f"{measure.__class__.__name__}:")
        print(f"    Similar Grid:", measure.get_similarity(grid1, grid2))
        print(f"    Different Grid:", measure.get_similarity(grid1, grid3))

if __name__ == "__main__":
    main()