import random

import numpy as np

from compactreasoningmodels.utils.grid import derive_clues_from_grid


class ClueGenerator:
    def __init__(self, w: int, h: int, prob: float):
        self.w = w
        self.h = h
        self.prob = prob
        self.memo: set[tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]] = set()

    def gen_grid(self) -> list[list[int]]:
        return [
            [1 if random.random() < self.prob else 0 for _ in range(self.w)] for _ in range(self.h)
        ]

    def find_clues_from_grid(
        self, grid: list[list[int]]
    ) -> tuple[list[list[int]], list[list[int]]]:
        return derive_clues_from_grid(grid)

    def gen_clues_and_grid(
        self, max_attempts: int = 100
    ) -> tuple[tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]], np.ndarray]:
        for _ in range(max_attempts):
            grid = self.gen_grid()
            clues = self.find_clues_from_grid(grid)
            tuple_clues: tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]] = (
                tuple(tuple(row) for row in clues[0]),
                tuple(tuple(col) for col in clues[1]),
            )
            if tuple_clues not in self.memo:
                self.memo.add(tuple_clues)
                return tuple_clues, np.array(grid)
        raise RuntimeError(f"Failed to generate unique clues after {max_attempts} attempts")

    def gen_clues(
        self, max_attempts: int = 100
    ) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
        for _ in range(max_attempts):
            grid = self.gen_grid()
            row_clues, col_clues = self.find_clues_from_grid(grid)
            tuple_clues = (
                tuple(tuple(row) for row in row_clues),
                tuple(tuple(col) for col in col_clues),
            )
            if tuple_clues not in self.memo:
                self.memo.add(tuple_clues)
                return tuple_clues
        raise RuntimeError(f"Failed to generate unique clues after {max_attempts} attempts")
