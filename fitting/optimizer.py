class Optimizer():
    '''Optimizer class '''

    def __init__(self, name, display_graphics):
        self.name = name
        self.display_graphics = display_graphics
        self.optimized_variables = []
        self.goodness_of_fit = []
        
    def get_fit(self, fit_function, **kwargs):
        return (fit_function)(self.optimized_variables, **kwargs)