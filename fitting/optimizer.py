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
        ''' Sets a fit function '''
        self.fit_function = func
    
    def set_objective_function(self, func):
        ''' Sets an objective function '''
        self.objective_function = func
    
    def get_fit(self):
        ''' Gets fits '''
        return (self.fit_function)(self.optimized_variables)