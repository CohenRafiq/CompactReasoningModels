from typing import List, Tuple
import numpy as np
import torch
from ..utils.types import ShapeError
from ..trace_comparison.blocks.base import Block

class Experiment:
    
    def __init__(self, blocks: List[Block] = None, name: str = "experiment"):
        self.blocks: List[Block] = []
        self.name = name
        self._shape_validated = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if blocks:
            for block in blocks:
                self.add_block(block)
    
    def add_block(self, block: Block) -> 'Experiment':
        self.blocks.append(block)
        self._shape_validated = False
        return self
    
    def validate_shapes(self, input_shape: Tuple[int, ...]) -> None:
        # (solvers, batch, width, height, steps)
        if len(input_shape) != 5:
            raise ShapeError(f"Input shape must be 5D (solvers, batch, width, height, steps), got {len(input_shape)}D")

        current_shape = input_shape
        for i, block in enumerate(self.blocks):
            try:
                next_shape = block.compute_output_shape(current_shape)
                current_shape = next_shape
            except ShapeError as e:
                raise ShapeError(
                    f"Shape validation failed at block {i+1} "
                    f"('{block.name}'): {str(e)}"
                )
        if current_shape != (1,):
            raise ShapeError(f"Final output shape must be scalar (1,), got {current_shape}")
        self._shape_validated = True
    
    def run(self, data: torch.Tensor | np.ndarray, validate: bool = True) -> torch.Tensor:
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, device=self.device)
        if not isinstance(data, torch.Tensor):
            raise TypeError(f"Input data must be a torch.Tensor or np.ndarray, got {type(data)}")
        data = data.to(self.device)
        if validate and not self._shape_validated:
            self.validate_shapes(data.shape)
        
        current_data = data
        for block in self.blocks:
            current_data = block(current_data)
        
        return current_data
    
    def __call__(self, data: torch.Tensor | np.ndarray) -> torch.Tensor:
        return self.run(data)
        
    def __repr__(self):
        return f"Experiment(name='{self.name}', blocks={len(self.blocks)})"