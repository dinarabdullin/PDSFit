import numpy as np
from functools import partial
from background.background import Background


def background_model(t, decay_constant):
    return np.exp(-decay_constant * np.abs(t))

def signal_model(t, decay_constant, scale_factor, s_intra):
    return background_model(t, decay_constant) * (np.ones(s_intra.size) + scale_factor * (s_intra - np.ones(s_intra.size)))

def signal_model_wrapper(t, scale_factor, decay_constant, s_intra):
    return model(t, decay_constant, scale_factor, s_intra)


class ExponentialBackground(Background):
    ''' Exponential background '''
    
    def __init__(self):
        super().__init__() 
        self.parameter_names = ['decay_constant', 'scale_factor']
        self.parameter_full_names = {'decay_constant': 'Decay constant' , 'scale_factor': 'Scale factor'}
      
    def set_fit_function(self, s_intra):   
        if self.parameters['decay_constant']['optimize'] and self.parameters['scale_factor']['optimize']:
            self.fit_function = partial(signal_model, 
                                        s_intra=s_intra)
        elif self.parameters['decay_constant']['optimize'] and not self.parameters['scale_factor']['optimize']:
            self.fit_function = partial(signal_model, 
                                        scale_factor=self.parameters['scale_factor']['value'],
                                        s_intra=s_intra)
        elif not self.parameters['decay_constant']['optimize'] and self.parameters['scale_factor']['optimize']:
            self.fit_function = partial(signal_model_wrapper, 
                                        decay_constant=self.parameters['decay_constant']['value'],
                                        s_intra=s_intra)
        elif not self.parameters['decay_constant']['optimize'] and not self.parameters['scale_factor']['optimize']:
            self.fit_function = partial(signal_model, 
                                        decay_constant = self.parameters['decay_constant']['value'],
                                        scale_factor=self.parameters['scale_factor']['value'],
                                        s_intra=s_intra)
    
    def get_fit(self, t, background_parameters, s_intra):
        return signal_model(t, background_parameters['decay_constant'], background_parameters['scale_factor'], s_intra)
    
    def get_background(self, t, background_parameters, modulation_depth=0):
        return background_model(t, background_parameters['decay_constant']) * (1-background_parameters['scale_factor']*modulation_depth)