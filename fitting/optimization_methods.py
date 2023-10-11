"""The list of supported optimization algorithms."""
from fitting.optimizer import Optimizer
from fitting.ga.ga import GeneticAlgorithm


optimization_methods = {}
optimization_methods["ga"] = GeneticAlgorithm