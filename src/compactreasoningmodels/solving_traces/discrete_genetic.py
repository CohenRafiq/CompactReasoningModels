import numpy as np
import torch

from compactreasoningmodels.solving_traces.base import SolvingTrace

# Simulates population and evolves it using
# genetic algorithm to minimize constraint violations
# Uses simple mu lambda algorithm, fully vectorized (no per-individual
# Python loops) so it scales to large population_size / concurrent_samples.


class DiscreteGeneticAlgorithm(SolvingTrace):
    def __init__(
        self,
        clues: np.ndarray | torch.Tensor,
        grid_shape: tuple[int, int],
        initial_grid: np.ndarray | torch.Tensor | None = None,
        population_size: int = 100,
        concurrent_samples: int = 100,
        mutation_rate: float = 0.01,
        tournament_size: int = 3,
    ):
        super().__init__(clues, grid_shape, initial_grid)
        self.population_size = population_size
        self.concurrent_samples = concurrent_samples
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.grid_shape = grid_shape
        self.rows, self.cols = grid_shape

        self.row_clues = np.asarray(self.clues[0])  # (rows, K)
        self.col_clues = np.asarray(self.clues[1])  # (cols, K)
        self.K_row = self.row_clues.shape[1]
        self.K_col = self.col_clues.shape[1]
        self.row_nnz = (self.row_clues > 0).sum(axis=1)  # (rows,) expected run count per row
        self.col_nnz = (self.col_clues > 0).sum(axis=1)  # (cols,)

        self.population = self.random_population(
            self.population_size * self.concurrent_samples, grid_shape
        )

    @staticmethod
    def random_population(population_size: int, grid_shape: tuple[int, int]) -> np.ndarray:
        return np.random.randint(0, 2, size=(population_size, *grid_shape)).astype(np.int8)

    @staticmethod
    def _line_run_lengths(lines: np.ndarray, K: int) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized run-length extraction for a batch of binary lines."""
        from compactreasoningmodels.utils.grid import batch_line_clues

        return batch_line_clues(lines, K)

    def count_violations_batch(self, grids: np.ndarray) -> np.ndarray:
        """Vectorized violation count for a whole batch of grids at once.

        grids: (M, rows, cols) int array of 0/1.
        Returns: (M,) violation counts.
        """
        M = grids.shape[0]

        # --- rows ---
        row_lines = grids.reshape(M * self.rows, self.cols)
        row_runs, row_num_runs = self._line_run_lengths(row_lines, self.K_row)
        row_clue_tiled = np.tile(self.row_clues, (M, 1))  # (M*rows, K_row)
        row_nnz_tiled = np.tile(self.row_nnz, M)  # (M*rows,)
        row_ok = np.all(row_runs == row_clue_tiled, axis=1) & (row_num_runs == row_nnz_tiled)
        row_violations = (~row_ok).reshape(M, self.rows).sum(axis=1)

        # --- columns ---
        col_lines = grids.transpose(0, 2, 1).reshape(M * self.cols, self.rows)
        col_runs, col_num_runs = self._line_run_lengths(col_lines, self.K_col)
        col_clue_tiled = np.tile(self.col_clues, (M, 1))  # (M*cols, K_col)
        col_nnz_tiled = np.tile(self.col_nnz, M)  # (M*cols,)
        col_ok = np.all(col_runs == col_clue_tiled, axis=1) & (col_num_runs == col_nnz_tiled)
        col_violations = (~col_ok).reshape(M, self.cols).sum(axis=1)

        return row_violations + col_violations

    def count_violations(self, grid: np.ndarray) -> int:
        """Single-grid convenience wrapper (kept for API compatibility)."""
        return int(self.count_violations_batch(grid[None, ...])[0])

    def fitness_batch(self, grids: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + self.count_violations_batch(grids))

    def fitness(self, grid: np.ndarray) -> float:
        return float(self.fitness_batch(grid[None, ...])[0])

    def heatmap_step(self, grid: np.ndarray) -> np.ndarray:
        """
        Runs one (mu, lambda)-style generation of the GA across ALL
        concurrent_samples populations simultaneously (fully vectorized —
        no Python loop over individuals or groups), then returns a heatmap
        = the elementwise mean of the best individual from each concurrent
        population.

        `grid` is accepted for interface consistency with the other
        SolvingTrace subclasses but isn't otherwise used: this trace evolves
        its own internal discrete population rather than refining a
        probability grid directly.
        """
        S, mu = self.concurrent_samples, self.population_size
        pop = self.population.reshape(S, mu, self.rows, self.cols)

        fitness = self.fitness_batch(self.population).reshape(S, mu)

        # --- elitism ---
        elite_idx = np.argmax(fitness, axis=1)  # (S,)
        s_range = np.arange(S)
        elite = pop[s_range, elite_idx]  # (S, rows, cols)

        # --- vectorized tournament selection for the remaining mu-1 slots ---
        num_children = mu - 1
        t = self.tournament_size

        def select_parents() -> np.ndarray:
            contenders = np.random.randint(0, mu, size=(S, num_children, t))
            fit_c = fitness[s_range[:, None, None], contenders]
            winner_local = np.argmax(fit_c, axis=2)
            winner_idx = np.take_along_axis(contenders, winner_local[..., None], axis=2)
            return winner_idx.squeeze(-1)

        parent1_idx = select_parents()
        parent2_idx = select_parents()

        parents1 = pop[s_range[:, None], parent1_idx]  # (S, C, rows, cols)
        parents2 = pop[s_range[:, None], parent2_idx]  # (S, C, rows, cols)

        # --- uniform crossover ---
        cross_mask = np.random.randint(0, 2, size=parents1.shape).astype(bool)
        children = np.where(cross_mask, parents1, parents2)

        # --- mutation ---
        flip_mask = np.random.random(size=children.shape) < self.mutation_rate
        children = np.where(flip_mask, 1 - children, children).astype(np.int8)

        new_pop = np.concatenate([elite[:, None, :, :], children], axis=1)  # (S, mu, rows, cols)
        self.population = new_pop.reshape(S * mu, self.rows, self.cols)

        return elite.mean(axis=0).astype(np.float64)


if __name__ == "__main__":
    clues = np.array(
        [
            [[0, 0, 0], [3, 0, 0], [1, 1, 0], [3, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [3, 0, 0], [1, 1, 0], [3, 0, 0], [0, 0, 0]],
        ]
    )
    solver = DiscreteGeneticAlgorithm(clues, (5, 5))
    grid = np.full((5, 5), 0.5)
    print(clues.shape)
    print(grid.shape)
    for _ in range(10):
        grid = solver.heatmap_step(grid)
        print(grid)
