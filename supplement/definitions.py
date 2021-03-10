''' Constants & definitions '''

import numpy as np

const = {}
const['Hz2MHz'] = 1e-6
const['MHz2Hz'] = 1e6
const['MHz2GHz'] = 1e-3
const['GHz2MHz'] = 1e3
const['mT2T'] = 1e-3
const['T2mT'] = 1e3
const['nm2m'] = 1e-9
const['deg2rad'] = np.pi / 180.0
const['rad2deg'] = 180.0 / np.pi
const['wn2MHz'] = 29979.0
const['plank_constant'] = 6.626070040e-34 # J*s
const['bohr_magneton'] = 9.274009994e-24 # J/T
const['bolzmann_constant'] = 1.38064852e-23 # J/K
const['ge'] = 2.0023 # free electron g factor
const['vacuum_permeability'] = 1e-7 # T*m/A
const['Fez'] = 1e-3 * const['Hz2MHz'] * const['bohr_magneton'] / const['plank_constant'] # GHz/T
const['Fdd'] = const['Hz2MHz'] * const['vacuum_permeability'] * const['bohr_magneton']**2 / (const['plank_constant'] * const['nm2m']**3) # MHz
const['fwhm2sd'] = 1 / 2.35482 # half width at half maximum -> standard deviation for the Gaussian function


# The intensities of spectral components for 
# nuclear spin I and the number of equivalent nuclei n
const['relative_intensities'] = [  
    #I = 1/2
    [[1, 1],                    #n = 1
    [1, 2, 1],                  #n = 2
    [1, 3, 3, 1],               #n = 3
    [1, 4, 6, 4, 1],            #n = 4
    [1, 5, 10, 10, 5, 1],       #n = 5
    [1, 6, 15, 20, 15, 6, 1]],  #n = 6
    #I = 1
    [[1, 1, 1],                                             #n = 1
    [1, 2, 3, 2, 1],                                        #n = 2
    [1, 3, 6, 7, 6, 3, 1],                                  #n = 3
    [1, 4, 10, 16, 19, 16, 10, 4, 1],                       #n = 4
    [1, 5, 15, 30, 45, 51, 45, 30, 15, 5, 1],               #n = 5
    [1, 6, 21, 50, 90, 126, 141, 126, 90, 50, 21, 6, 1]],   #n = 6
    #I = 3/2
	[[1, 1, 1, 1],                                                                          #n = 1
	[1, 2, 3, 4, 3, 2, 1],                                                                  #n = 2
	[1, 3, 6, 10, 12, 12, 10, 6, 3, 1],                                                     #n = 3
	[1, 4, 10, 20, 31, 40, 44, 40, 31, 20, 10, 4, 1],                                       #n = 4
	[1, 5, 15, 35, 65, 101, 135, 155, 155, 135, 101, 65, 35, 15, 5, 1],                     #n = 5
	[1, 6, 21, 56, 120, 216, 336, 456, 546, 580, 546, 456, 336, 216, 120, 56, 21, 6, 1]],   #n = 6
]

# Supported distributions
const['distribution_types'] = ['uniform', 'normal', 'vonmises']

# Supported Euler angles conventions
const['euler_angles_conventions'] = ['ZXZ', 'XYX', 'YZY', 'ZYZ', 'XZX', 'YXY']

