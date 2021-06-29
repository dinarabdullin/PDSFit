''' A dictionary of supported simulators '''

from simulation.simulator import Simulator
from simulation.monte_carlo_simulator import MonteCarloSimulator
#from simulation.grid_simulator import GridSimulator

simulator_types = {}
simulator_types['monte_carlo'] = MonteCarloSimulator
#simulator_types['grids'] = GridSimulator