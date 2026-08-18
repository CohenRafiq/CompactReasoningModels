import random
from itertools import groupby
from typing import List, Tuple
import numpy as np


class ClueGenerator:

    def __init__(self, w: int, h: int, prob: float):
        self.w = w
        self.h = h
        self.prob = prob
        self.memo = set()

    def gen_grid(self) -> List[List[int]]:
        return [[1 if random.random() < self.prob else 0 for _ in range(self.w)] for _ in range(self.h)]

    def _clue_line(self, line: List[int]) -> List[int]:
        return [sum(1 for _ in group) for key, group in groupby(line) if key == 1]

    def find_clues_from_grid(self, grid: List[List[int]]) -> Tuple[List[List[int]], List[List[int]]]:
        row_clues = [self._clue_line(row) for row in grid]
        col_clues = [self._clue_line(col) for col in zip(*grid)]
        return row_clues, col_clues

    def gen_clues_and_grid(self, max_attempts: int = 100) -> Tuple[List[List[int]], List[List[int]]]:
        for _ in range(max_attempts):
            grid = self.gen_grid()
            clues = self.find_clues_from_grid(grid)
            tuple_clues = (tuple(tuple(row) for row in clues[0]), tuple(tuple(col) for col in clues[1]))
            if tuple_clues not in self.memo:
                self.memo.add(tuple_clues)
                return tuple_clues, np.array(grid)

        