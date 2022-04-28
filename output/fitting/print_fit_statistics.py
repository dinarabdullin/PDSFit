import sys
import numpy as np
from supplement.definitions import const


def print_fit_statistics(fit_statistics):
    ''' Prints the statistics of the fitting '''
    sys.stdout.write('\nStatistics:\n')
    sys.stdout.write('{:<32}{:<20}\n'.format('Chi-squared: ', fit_statistics['chi2']))
    sys.stdout.write('{:<32}{:<20}\n'.format('Reduced chi-squared: ', fit_statistics['reduced_chi2']))
    sys.stdout.write('{:<32}{:<20}\n'.format('Akaike information criterion: ', fit_statistics['aic']))
    sys.stdout.write('{:<32}{:<20}\n'.format('Bayesian information criterion: ', fit_statistics['bic']))