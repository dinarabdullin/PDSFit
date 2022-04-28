import numpy as np
from mathematics.chi2 import chi2
from supplement.definitions import const


class Optimizer():
    '''Optimizer class '''

    def __init__(self, name, display_graphics, goodness_of_fit):
        self.name = name
        self.display_graphics = display_graphics
        self.goodness_of_fit = goodness_of_fit
        self.goodness_of_fit_name = const['goodness_of_fit_names'][goodness_of_fit]
        self.fit_function = None
        self.objective_function = None
        self.optimized_variables = []
        self.score = []
    
    def set_fit_function(self, func):
        ''' Sets the fit function '''
        self.fit_function = func
    
    def set_objective_function(self, func):
        ''' Sets the objective function '''
        self.objective_function = func
    
    def get_fit(self):
        ''' Calculates the fit to the PDS time traces '''
        return (self.fit_function)(self.optimized_variables)
    
    def get_fit_statistics(self, experiments, simulated_time_traces=[]):
        ''' Calculate the statistics describing the goodness of fit '''
        fit_statistics = {}
        if simulated_time_traces == []:
            simulated_time_traces, background_parameters, background_time_traces = self.get_fit()
        # Calculate the number of points, the number of fitting parameters, and the number of degrees of freedom
        N = 0
        for i in range(len(experiments)):
            N += simulated_time_traces[i]['s'].size
        q = self.optimized_variables.size
        K = q + 1
        Nf = float(N)
        qf = float(q)
        Kf = float(K)
        # Chi2 and reduced Chi2
        chi2_value = 0
        for i in range(len(experiments)):
            chi2_value += chi2(simulated_time_traces[i]['s'], experiments[i].s, experiments[i].noise_std)
        reduced_chi2_value = chi2_value / (Nf - qf)
        fit_statistics['chi2'] = chi2_value
        fit_statistics['reduced_chi2'] = reduced_chi2_value
        # Akaike information criterion and Bayesian information criterion
        msd = 0
        for i in range(len(experiments)):
            msd += np.sum((simulated_time_traces[i]['s'] - experiments[i].s)**2)
        msd = msd / Nf
        aic_value = Nf * np.log(msd) + 2 * Kf + 2 * Kf * (Kf + 1) / (Nf - Kf - 1)
        bic_value = Nf * np.log(msd) + Kf * np.log(Nf)
        fit_statistics['aic'] = aic_value
        fit_statistics['bic'] = bic_value
        return fit_statistics 