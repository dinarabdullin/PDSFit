''' A dictionary of supported optimization algorithms '''

from fitting.optimizer import Optimizer
from fitting.ga.ga import GeneticAlgorithm
from fitting.ga_local.ga_local import GeneticAlgorithmWithLocalSolver

optimization_methods = {}
optimization_methods['ga'] = GeneticAlgorithm
optimization_methods['ga_local'] = GeneticAlgorithmWithLocalSolver