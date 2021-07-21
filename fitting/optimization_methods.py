''' A dictionary of supported optimization algorithms '''

from fitting.optimizer import Optimizer
from fitting.ga.ga import GeneticAlgorithm
from fitting.ga_local.ga_local import GeneticAlgorithmWithLocalSolver
from fitting.ga_sa_local.ga_sa_local import GeneticAlgorithmWithSimulatedAnnealingAndLocalSolver

optimization_methods = {}
optimization_methods['ga'] = GeneticAlgorithm
optimization_methods['ga_local'] = GeneticAlgorithmWithLocalSolver
optimization_methods['ga_sa_local'] = GeneticAlgorithmWithSimulatedAnnealingAndLocalSolver