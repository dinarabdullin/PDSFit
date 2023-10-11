"""The list of supported simulators."""
from simulation.simulator import Simulator
from simulation.monte_carlo_simulator import MonteCarloSimulator


simulator_types = {}
simulator_types["monte_carlo"] = MonteCarloSimulator