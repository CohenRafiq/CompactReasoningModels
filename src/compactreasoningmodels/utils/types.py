from typing import NamedTuple


class SolverProfile(NamedTuple):
    name: str
    num_steps: int


class ShapeError(Exception):
    pass
