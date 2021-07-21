import numpy as np
from functools import partial
from background.background import Background


def model(t, decay_constant, dimension, scale_factor, s_intra):
    return np.exp(-decay_constant * np.abs(t)**(dimension/3.0)) * (np.ones(s_intra.size) + scale_factor * (s_intra - np.ones(s_intra.size)))

def model_wrapper1(t, dimension, scale_factor, decay_constant, s_intra):
    return model(t, decay_constant, dimension, scale_factor, s_intra)

def model_wrapper2(t, scale_factor, decay_constant, dimension, s_intra):
    return model(t, decay_constant, dimension, scale_factor, s_intra)
    

class StretchedExponentialBackground(Background):
    ''' Stretched exponential background '''
    
    def __init__(self):
        super().__init__() 
        self.parameter_names = ['decay_constant', 'dimension', 'scale_factor']
        self.parameter_full_names = {'decay_constant': 'Decay constant' , 'dimension': 'Dimension', 'scale_factor': 'Scale factor'}
      
    def set_fit_function(self, s_intra):   
        if self.parameters['decay_constant']['optimize'] and self.parameters['dimension']['optimize'] and self.parameters['scale_factor']['optimize']:
            self.fit_function = partial(model, 
                                        s_intra=s_intra)
        elif self.parameters['decay_constant']['optimize'] and self.parameters['dimension']['optimize'] and not self.parameters['scale_factor']['optimize']:
            self.fit_function = partial(model, 
                                        scale_factor=self.parameters['scale_factor']['value'],
                                        s_intra=s_intra)
        elif self.parameters['decay_constant']['optimize'] and not self.parameters['dimension']['optimize'] and not self.parameters['scale_factor']['optimize']:
            self.fit_function = partial(model, 
                                        dimension=self.parameters['dimension']['value'],
                                        scale_factor=self.parameters['scale_factor']['value'],
                                        s_intra=s_intra)
        elif not self.parameters['decay_constant']['optimize'] and self.parameters['dimension']['optimize'] and not self.parameters['scale_factor']['optimize']:
            self.fit_function = partial(model_wrapper1, 
                                        scale_factor=self.parameters['scale_factor']['value'],
                                        decay_constant=self.parameters['decay_constant']['value'],
                                        s_intra=s_intra)
        elif not self.parameters['decay_constant']['optimize'] and not self.parameters['dimension']['optimize'] and not self.parameters['scale_factor']['optimize']:
            self.fit_function = partial(model, 
                                        decay_constant=self.parameters['decay_constant']['value'],
                                        dimension=self.parameters['dimension']['value'],
                                        scale_factor=self.parameters['scale_factor']['value'],
                                        s_intra=s_intra)
        elif self.parameters['decay_constant']['optimize'] and not self.parameters['dimension']['optimize'] and self.parameters['scale_factor']['optimize']:
            self.fit_function = partial(model_wrapper2, 
                                        dimension=self.parameters['dimension']['value'],
                                        s_intra=s_intra)
        elif not self.parameters['decay_constant']['optimize'] and not self.parameters['dimension']['optimize'] and self.parameters['scale_factor']['optimize']:
            self.fit_function = partial(model_wrapper2, 
                                        decay_constant=self.parameters['decay_constant']['value'],
                                        dimension=self.parameters['dimension']['value'],
                                        s_intra=s_intra)
        elif not self.parameters['decay_constant']['optimize'] and self.parameters['dimension']['optimize'] and self.parameters['scale_factor']['optimize']:
            self.fit_function = partial(model_wrapper1,
                                        decay_constant=self.parameters['decay_constant']['value'],
                                        s_intra=s_intra)
    
    def get_fit(self, t, background_parameters, s_intra):
        return model(t, background_parameters['decay_constant'], background_parameters['dimension'], background_parameters['scale_factor'], s_intra)