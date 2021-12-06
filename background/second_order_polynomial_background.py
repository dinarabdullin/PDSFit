import numpy as np
from functools import partial
from background.background import Background


def background_model(t, c1, c2):
    return (1.0 + c1 * np.abs(t) + c2 * np.abs(t)**2)
    
def signal_model(t, c1, c2, scale_factor, s_intra):
    return background_model(t, c1, c2) * (np.ones(s_intra.size) + scale_factor * (s_intra - np.ones(s_intra.size)))

def signal_model_wrapper(t, scale_factor, c1, c2, s_intra):
    return signal_model(t, c1, c2, scale_factor, s_intra)
    

class SecondOrderPolynomialBackground(Background):
    ''' Second-order polynomial background '''
    
    def __init__(self):
        super().__init__() 
        self.parameter_names = ['c1', 'c2', 'scale_factor']
        self.parameter_full_names = {'c1': 'c1', 'c2': 'c2', 'scale_factor': 'Scale factor'}
      
    def set_fit_function(self, s_intra):   
        if self.parameters['c1']['optimize'] and self.parameters['c2']['optimize'] and self.parameters['scale_factor']['optimize']:
            self.fit_function = partial(signal_model, 
                                        s_intra=s_intra)
        elif self.parameters['c1']['optimize'] and self.parameters['c2']['optimize'] and not self.parameters['scale_factor']['optimize']:
            self.fit_function = partial(signal_model,
                                        scale_factor=self.parameters['scale_factor']['value'],            
                                        s_intra=s_intra)                                
        elif not self.parameters['c1']['optimize'] and not self.parameters['c2']['optimize'] and self.parameters['scale_factor']['optimize']:
            self.fit_function = partial(signal_model_wrapper, 
                                        c1=self.parameters['c1']['value'],
                                        c2=self.parameters['c2']['value'],
                                        s_intra=s_intra)                                
        elif not self.parameters['c1']['optimize'] and not self.parameters['c2']['optimize'] and not self.parameters['scale_factor']['optimize']:
            self.fit_function = partial(signal_model, 
                                        c1=self.parameters['c1']['value'],
                                        c2=self.parameters['c2']['value'],
                                        scale_factor=self.parameters['scale_factor']['value'],
                                        s_intra=s_intra)                                
        else:
            raise ValueError('The polynomial coefficient of the background can not be optimized separately!')
            sys.exit(1)
    
    def get_fit(self, t, background_parameters, s_intra):
        return signal_model(t, background_parameters['c1'], background_parameters['c2'], background_parameters['scale_factor'], s_intra)
    
    def get_background(self, t, background_parameters, modulation_depth=0):
        return background_model(t, background_parameters['c1'], background_parameters['c2']) * (1-background_parameters['scale_factor']*modulation_depth)