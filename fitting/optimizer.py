import numpy as np
from supplement.definitions import const


class Optimizer():
    '''Optimizer '''

    def __init__(self, name):
        self.name = name
        self.goodness_of_fit = None
        self.goodness_of_fit_name = None
        self.fit_function = None
        self.fit_function_more_output = None
        self.objective_function = None
        self.optimized_variables = []
        self.score = []
        self.idx_best_solution = 0
        
    def set_goodness_of_fit(self, goodness_of_fit):
        self.goodness_of_fit = goodness_of_fit
        self.goodness_of_fit_name = const['goodness_of_fit_names'][goodness_of_fit]
        
    def set_fit_function(self, func):
        ''' Sets the fit function '''
        self.fit_function = func
    
    def set_fit_function_more_output(self, func):
        ''' Sets the fit function with the extended output '''
        self.fit_function_more_output = func
    
    def set_objective_function(self, func):
        ''' Sets the objective function '''
        self.objective_function = func
    
    def get_fit(self):
        ''' Calculates the fit to the PDS time traces '''
        return (self.fit_function)(self.optimized_variables[self.idx_best_solution])
        
    def get_fit_more_output(self):
        ''' Calculates the fit to the PDS time traces and more'''
        return (self.fit_function_more_output)(self.optimized_variables[self.idx_best_solution])     