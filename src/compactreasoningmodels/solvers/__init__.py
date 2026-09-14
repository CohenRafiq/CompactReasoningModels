from compactreasoningmodels.solvers.base import BaseSolver
from compactreasoningmodels.solvers.mac import MAC
from compactreasoningmodels.solvers.genetic_algorithm import GeneticAlgorithmDET, GeneticAlgorithmDEP
from compactreasoningmodels.solvers.gradient_descent import (
    GDGlobalAdamSolver, GDGlobalSGDSolver,
    GDGaussSeidelAdamSolver, GDGaussSeidelSGDSolver,
    GDJacobiAdamSolver, GDJacobiSGDSolver,
    GDBangBangSolver,
)
from compactreasoningmodels.solvers.search import BacktrackingSearch
from compactreasoningmodels.solvers.model_solver import ModelSolver


__all__ = [
    "BaseSolver",
    "MAC",
    "GeneticAlgorithmDET",
    "GeneticAlgorithmDEP",
    "GDGlobalAdamSolver",
    "GDGlobalSGDSolver",
    "GDGaussSeidelAdamSolver",
    "GDGaussSeidelSGDSolver",
    "GDJacobiAdamSolver",
    "GDJacobiSGDSolver",
    "GDBangBangSolver",
    "BacktrackingSearch",
    "ModelSolver",
]

SOLVERS = {
    "mac": MAC,
    "genetic_algorithm_det": GeneticAlgorithmDET,
    "genetic_algorithm_dep": GeneticAlgorithmDEP,
    "gradient_descent_global_adam": GDGlobalAdamSolver,
    "gradient_descent_global_sgd": GDGlobalSGDSolver,
    "gradient_descent_bang_bang": GDBangBangSolver,
    # "gradient_descent_gauss_seidel_adam": GDGaussSeidelAdamSolver,
    # "gradient_descent_gauss_seidel_sgd": GDGaussSeidelSGDSolver,
    # "gradient_descent_jacobi_adam": GDJacobiAdamSolver,
    # "gradient_descent_jacobi_sgd": GDJacobiSGDSolver,
    "model_solver": ModelSolver,
    "backtracking_search": BacktrackingSearch,
}