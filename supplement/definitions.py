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
const['ns2us'] = 1e-3
const['deg2rad'] = np.pi / 180.0
const['rad2deg'] = 180.0 / np.pi
const['plank_constant'] = 6.626070040e-34 # J*s
const['bohr_magneton'] = 9.274009994e-24 # J/T
const['bolzmann_constant'] = 1.38064852e-23 # J/K
const['ge'] = 2.0023 # free electron g factor
const['vacuum_permeability'] = 1e-7 # T*m/A
const['Fez'] = 1e-3 * const['Hz2MHz'] * const['bohr_magneton'] / const['plank_constant'] # GHz/T
const['Fdd'] = const['Hz2MHz'] * const['vacuum_permeability'] * const['bohr_magneton']**2 / (const['plank_constant'] * const['nm2m']**3) # MHz
const['fwhm2sd'] = 1 / 2.35482 # half width at half maximum -> standard deviation for the Gaussian function
const['pp2sd'] = 0.5 # peak-to-peak width -> standard deviation for the Gaussian function


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

# Names of fitting parameters
const['fitting_parameters_names'] = [
    'r_mean',
    'r_width', 
    'xi_mean', 
    'xi_width', 
    'phi_mean', 
    'phi_width',
    'alpha_mean', 
    'alpha_width', 
    'beta_mean', 
    'beta_width',
    'gamma_mean', 
    'gamma_width', 
    'rel_prob',
    'j_mean', 
    'j_width'
    ]

# Names of paired fitting parameters
const['paired_fitting_parameters'] = {
    'r_mean'      : 'r_width',
    'r_width'     : 'r_mean', 
    'xi_mean'     : 'xi_width',
    'xi_width'    : 'xi_mean', 
    'phi_mean'    : 'phi_width', 
    'phi_width'   : 'phi_mean',
    'alpha_mean'  : 'alpha_width',
    'alpha_width' : 'alpha_mean',
    'beta_mean'   : 'beta_width',
    'beta_width'  : 'beta_mean',
    'gamma_mean'  : 'gamma_width',
    'gamma_width' : 'gamma_mean',
    'rel_prob'    : 'none',
    'j_mean'      : 'j_width', 
    'j_width'     : 'j_mean'
    }
 
# Names of angle parameters
const['angle_parameters_names'] = [
    'xi_mean',  
    'phi_mean', 
    'alpha_mean', 
    'beta_mean',
    'gamma_mean',
    ]

# Scale factors for fitting parameters
const['fitting_parameters_scales'] = {
    'r_mean'      : 1.0,
    'r_width'     : 1.0, 
    'xi_mean'     : const['deg2rad'],
    'xi_width'    : const['deg2rad'], 
    'phi_mean'    : const['deg2rad'], 
    'phi_width'   : const['deg2rad'],
    'alpha_mean'  : const['deg2rad'],
    'alpha_width' : const['deg2rad'],  
    'beta_mean'   : const['deg2rad'], 
    'beta_width'  : const['deg2rad'],
    'gamma_mean'  : const['deg2rad'], 
    'gamma_width' : const['deg2rad'],
    'rel_prob'    : 1.0,
    'j_mean'      : 1.0, 
    'j_width'     : 1.0
    }
    
# Long names of fitting parameters    
const['fitting_parameters_names_and_units'] = {
	'r_mean'      : 'r mean (nm)',
	'r_width'     : 'r width (nm)', 
	'xi_mean'     : 'xi mean (deg)',
	'xi_width'    : 'xi width (deg)', 
	'phi_mean'    : 'phi mean (deg)', 
	'phi_width'   : 'phi width (deg)',
    'alpha_mean'  : 'alpha mean (deg)',
    'alpha_width' : 'alpha width (deg)',
    'beta_mean'   : 'beta mean (deg)',
    'beta_width'  : 'beta width (deg)',
    'gamma_mean'  : 'gamma mean (deg)',
    'gamma_width' : 'gamma width (deg)',
    'rel_prob'    : 'rel_prob',
    'j_mean'      : 'J mean (MHz)', 
    'j_width'     : 'J width (MHz)'
    }

# Axes' labels of fitting parameters    
const['fitting_parameters_labels'] = {
	'r_mean'      : [r'$\langle\mathit{r}\rangle$', '(nm)'],
	'r_width'     : [r'$\mathit{\Delta r}$', '(nm)'], 
	'xi_mean'     : [r'$\langle\mathit{\xi}\rangle$', '$^\circ$'],
	'xi_width'    : [r'$\mathit{\Delta\xi}$', '$^\circ$'], 
	'phi_mean'    : [r'$\langle\mathit{\varphi}\rangle$', '$^\circ$'], 
	'phi_width'   : [r'$\mathit{\Delta\varphi}$', '$^\circ$'],
    'alpha_mean'  : [r'$\langle\mathit{\alpha}\rangle$', '$^\circ$'],
    'alpha_width' : [r'$\mathit{\Delta\alpha}$', '$^\circ$'], 
    'beta_mean'   : [r'$\langle\mathit{\beta}\rangle$', '$^\circ$'],
    'beta_width'  : [r'$\mathit{\Delta\beta}$', '$^\circ$'], 
    'gamma_mean'  : [r'$\langle\mathit{\gamma}\rangle$', '$^\circ$'],
    'gamma_width' : [r'$\mathit{\Delta\gamma}$', '$^\circ$'],
    'rel_prob'    : [r'Relative weight'],
    'j_mean'      : [r'$\langle\mathit{J}\rangle$', '(MHz)'],
    'j_width'     : [r'$\mathit{\Delta J}$', '(MHz)']
    }
    
# Long names of goodness-of-fit parameters 
const['goodness_of_fit_names'] = {
    'chi2':                              'Chi2',
    'reduced_chi2':                      'Reduced Chi2',
    'chi2_noise_std_1':                  'Chi2(noise_std=1)',
    }
    
# Axes' labels of goodness-of-fit parameters  
const['goodness_of_fit_axes_labels'] = {
    'chi2':                              r'$\mathit{\chi^2}$',
    'reduced_chi2':                      r'$\mathit{\chi^2_{\nu}}$',
    'chi2_noise_std_1':                  r'$\mathit{\chi^2}$ ($\mathit{\sigma_{n}}$ = 1)',
    } 