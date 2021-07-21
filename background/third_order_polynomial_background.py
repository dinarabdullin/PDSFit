import numpy as np
from functools import partial
from background.background import Background


def model(t, c0, c1, c2, c3, scale_factor, s_intra):
    return (c0 + c1 * np.abs(t) + c2 * np.abs(t)**2 + c3 * np.abs(t)**3) * (np.ones(s_intra.size) + scale_factor * (s_intra - np.ones(s_intra.size)))

def model_wrapper(t, scale_factor, c0, c1, c2, c3, s_intra):
    return model(t, c0, c1, c2, c3, scale_factor, s_intra)
    

class ThirdOrderPolynomialBackground(Background):
    ''' Third-order polynomial background '''
    
    def __init__(self):
        super().__init__() 
        self.parameter_names = ['c0', 'c1', 'c2', 'c3', 'scale_factor']
        self.parameter_full_names = {'c0': 'c0' , 'c1': 'c1', 'c2': 'c2', 'c3': 'c3', 'scale_factor': 'Scale factor'}
      
    def set_fit_function(self, s_intra):   
        if self.parameters['c0']['optimize'] and self.parameters['c1']['optimize'] and self.parameters['c2']['optimize'] and \
            self.parameters['c3']['optimize'] and self.parameters['scale_factor']['optimize']:
            self.fit_function = partial(model, 
                                        s_intra=s_intra)
        elif self.parameters['c0']['optimize'] and self.parameters['c1']['optimize'] and self.parameters['c2']['optimize'] and \
            self.parameters['c3']['optimize'] and not self.parameters['scale_factor']['optimize']:
            self.fit_function = partial(model,
                                        scale_factor=self.parameters['scale_factor']['value'],            
                                        s_intra=s_intra)                                
        elif not self.parameters['c0']['optimize'] and not self.parameters['c1']['optimize'] and not self.parameters['c2']['optimize'] and \
            not self.parameters['c3']['optimize'] and self.parameters['scale_factor']['optimize']:
            self.fit_function = partial(model_wrapper, 
                                        c0=self.parameters['c0']['value'],
                                        c1=self.parameters['c1']['value'],
                                        c2=self.parameters['c2']['value'],
                                        c3=self.parameters['c3']['value'],
                                        s_intra=s_intra)                                
        elif not self.parameters['c0']['optimize'] and not self.parameters['c1']['optimize'] and not self.parameters['c2']['optimize'] and \
            not self.parameters['c3']['optimize'] and not self.parameters['scale_factor']['optimize']:
            self.fit_function = partial(model, 
                                        c0=self.parameters['c0']['value'],
                                        c1=self.parameters['c1']['value'],
                                        c2=self.parameters['c2']['value'],
                                        c3=self.parameters['c3']['value'],
                                        scale_factor=self.parameters['scale_factor']['value'],
                                        s_intra=s_intra)                                
        else:
            raise ValueError('The polynomial coefficient of the background can not be optimized separately!')
            sys.exit(1)
    
    def get_fit(self, t, background_parameters, s_intra):
        return model(t, background_parameters['c0'], background_parameters['c1'], background_parameters['c2'], background_parameters['c3'], background_parameters['scale_factor'], s_intra)