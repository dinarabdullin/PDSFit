class ParameterID:
    ''' Identifier class of a model parameter ''' 

    def __init__(self, name, component):
        self.name = name
        self.component = component
    
    def get_index(self, fitting_parameters_indices):
        return fitting_parameters_indices[self.name][self.component].index
    
    def is_optimized(self, fitting_parameters_indices):
        return fitting_parameters_indices[self.name][self.component].optimize