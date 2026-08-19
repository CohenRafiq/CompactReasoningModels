from abc import ABC, abstractmethod

import numpy as np


class SimilarityMeasure(ABC):

    def get_similarity(self, grid1: list[list[int]], grid2: list[list[int]]) -> float:
        if len(grid1) != len(grid2) or len(grid1[0]) != len(grid2[0]):
            raise ValueError("Grid sizes must match for similarity computation.")
        return self.compute_similarity(grid1, grid2)

    @abstractmethod
    def compute_similarity(self, grid1: list[list[int]], grid2: list[list[int]]) -> float:
        ...

class MSE(SimilarityMeasure):
    def compute_similarity(self, grid1: list[list[int]], grid2: list[list[int]]) -> float:
        total_cells = len(grid1) * len(grid1[0])
        mse = sum((cell1 - cell2) ** 2 for row1, row2 in zip(grid1, grid2) for cell1, cell2 in zip(row1, row2)) / total_cells
        return mse

class MAE(SimilarityMeasure):
    def compute_similarity(self, grid1: list[list[int]], grid2: list[list[int]]) -> float:
        total_cells = len(grid1) * len(grid1[0])
        mae = sum(abs(cell1 - cell2) for row1, row2 in zip(grid1, grid2) for cell1, cell2 in zip(row1, row2)) / total_cells
        return mae

class HuberLoss(SimilarityMeasure):
    def __init__(self, delta: float = 1.0):
        self.delta = delta

    def compute_similarity(self, grid1: list[list[int]], grid2: list[list[int]]) -> float:
        total_cells = len(grid1) * len(grid1[0])
        huber_loss = 0.0
        for row1, row2 in zip(grid1, grid2):
            for cell1, cell2 in zip(row1, row2):
                diff = abs(cell1 - cell2)
                if diff <= self.delta:
                    huber_loss += 0.5 * diff ** 2
                else:
                    huber_loss += self.delta * (diff - 0.5 * self.delta)
        return huber_loss / total_cells

class MeanCosineSimilarity(SimilarityMeasure):
    def compute_similarity(self, grid1: list[list[int]], grid2: list[list[int]]) -> float:
        grid1_rows = [row for row in grid1]
        grid2_rows = [row for row in grid2]
        grid1_cols = [list(col) for col in zip(*grid1)]
        grid2_cols = [list(col) for col in zip(*grid2)]

        row_similarities = [self.cosine_similarity(r1, r2) for r1, r2 in zip(grid1_rows, grid2_rows)]
        col_similarities = [self.cosine_similarity(c1, c2) for c1, c2 in zip(grid1_cols, grid2_cols)]

        mean_row_similarity = sum(row_similarities) / len(row_similarities)
        mean_col_similarity = sum(col_similarities) / len(col_similarities)

        return 1 - (mean_row_similarity + mean_col_similarity) / 2.0   

    def cosine_similarity(self,vec1, vec2):
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a ** 2 for a in vec1) ** 0.5
        norm2 = sum(b ** 2 for b in vec2) ** 0.5
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

class PearsonCorrelation(SimilarityMeasure):
    def compute_similarity(self, grid1: list[list[int]], grid2: list[list[int]]) -> float:
        grid1_flat = np.array(grid1).flatten()
        grid2_flat = np.array(grid2).flatten()
        if len(grid1_flat) != len(grid2_flat):
            raise ValueError("Grid sizes must match for similarity computation.")
        if np.std(grid1_flat) == 0 or np.std(grid2_flat) == 0:
            return 0.0
        correlation_matrix = np.corrcoef(grid1_flat, grid2_flat)
        return correlation_matrix[0, 1]