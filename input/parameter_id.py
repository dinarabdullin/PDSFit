from input.parameter_object import ParameterObject


class ParameterID:
    ''' ID of a fitting parameter ''' 

    def __init__(self, name, spin_pair, component):
        self.name = name
        self.spin_pair = spin_pair
        self.component = component
    
    def get_index(self, fitting_parameters_indices):
        return fitting_parameters_indices[self.name][self.spin_pair][self.component].index
    
    def is_optimized(self, fitting_parameters_indices):
        return fitting_parameters_indices[self.name][self.spin_pair][self.component].optimize