"""
Inputs:
Solving trace 1: (B, H, W, steps_A)
Solving trace 2: (B, H, W, steps_B)

Output:
Similarity score: Scalar value [0-1]
"""

from compactreasoningmodels.trace_similarity.step_alignment import StepAlignment
import numpy as np


class TraceSimilarity:
    def __init__(self, step_alignment: StepAlignment, trace1: np.ndarray, trace2: np.ndarray):
        self.step_alignment = step_alignment

    def compute_similarity(self, trace1: np.ndarray, trace2: np.ndarray) -> float:
        aligned_trace1, aligned_trace2 = self.step_alignment.align(trace1, trace2)
        # Compute similarity between aligned traces (example: cosine similarity)
        dot_product = np.sum(aligned_trace1 * aligned_trace2)
        norm1 = np.linalg.norm(aligned_trace1)
        norm2 = np.linalg.norm(aligned_trace2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        similarity_score = dot_product / (norm1 * norm2)
        return similarity_score