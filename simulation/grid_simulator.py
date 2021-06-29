import sys
import time
import datetime
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import curve_fit
from functools import partial
from simulation.simulator import Simulator
from simulation.background_fit_function import *
from mathematics.random_points_on_sphere import random_points_on_sphere
from mathematics.random_points_from_distribution import random_points_from_distribution, random_points_from_sine_weighted_distribution
from mathematics.coordinate_system_conversions import spherical2cartesian, cartesian2spherical
from mathematics.rotate_coordinate_system import rotate_coordinate_system
from mathematics.histogram import histogram
from mathematics.chi2 import chi2
from mathematics.exponential_decay import exponential_decay
from supplement.definitions import const


class GridSimulator(Simulator):
    ''' Grid Simulation class '''
    
    def __init__(self, calculation_settings):
        super().__init__(calculation_settings)
        self.mc_sample_size = calculation_settings['mc_sample_size']
        self.separate_grids = True
        self.frequency_increment_epr_spectrum = 0.001 # in GHz
        self.frequency_increment_dipolar_spectrum = 0.01 # in MHz
        self.field_orientations = []
        self.effective_gfactors_spin1 = []
        self.detection_probabilities_spin1 = {}
        self.pump_probabilities_spin1 = {}