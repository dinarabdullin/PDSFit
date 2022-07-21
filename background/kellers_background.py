import numpy as np
from functools import partial
from background.background import Background


def background_model(t, k1, k2):
    return np.exp(k1 * np.abs(t) + k2 * t * t)

def signal_model(t, k1, k2, scale_factor, s_intra):
    return background_model(t, k1, k2) * (np.ones(s_intra.size) + scale_factor * (s_intra - np.ones(s_intra.size)))

def signal_model_wrapper1(t, k2, scale_factor, k1, s_intra):
    return signal_model(t, k1, k2, scale_factor, s_intra)

def signal_model_wrapper2(t, scale_factor, k1, k2, s_intra):
    return signal_model(t, k1, k2, scale_factor, s_intra)
    

class KellersBackground(Background):
    ''' RIDME background model proposed by Keller: a product of the exponential decay and a Gaussian '''
    
    def __init__(self):
        super().__init__() 
        self.parameter_names = ['k1', 'k2', 'scale_factor']
        self.parameter_full_names = {'k1': 'k1' , 'k2': 'k2', 'scale_factor': 'Scale factor'}
      
    def set_fit_function(self, s_intra):   
        if self.parameters['k1']['optimize'] and self.parameters['k2']['optimize'] and self.parameters['scale_factor']['optimize']:
            self.fit_function = partial(signal_model, 
                                        s_intra=s_intra)
        elif self.parameters['k1']['optimize'] and self.parameters['k2']['optimize'] and not self.parameters['scale_factor']['optimize']:
            self.fit_function = partial(signal_model, 
                                        scale_factor=self.parameters['scale_factor']['value'],
                                        s_intra=s_intra)
        elif self.parameters['k1']['optimize'] and not self.parameters['k2']['optimize'] and not self.parameters['scale_factor']['optimize']:
            self.fit_function = partial(signal_model, 
                                        k2=self.parameters['k2']['value'],
                                        scale_factor=self.parameters['scale_factor']['value'],
                                        s_intra=s_intra)
        elif not self.parameters['k1']['optimize'] and self.parameters['k2']['optimize'] and not self.parameters['scale_factor']['optimize']:
            self.fit_function = partial(signal_model_wrapper1, 
                                        scale_factor=self.parameters['scale_factor']['value'],
                                        k1=self.parameters['k1']['value'],
                                        s_intra=s_intra)
        elif not self.parameters['k1']['optimize'] and not self.parameters['k2']['optimize'] and not self.parameters['k3']['optimize']:
            self.fit_function = partial(signal_model, 
                                        k1=self.parameters['k1']['value'],
                                        k2=self.parameters['k2']['value'],
                                        scale_factor=self.parameters['scale_factor']['value'],
                                        s_intra=s_intra)
        elif self.parameters['k1']['optimize'] and not self.parameters['k2']['optimize'] and self.parameters['scale_factor']['optimize']:
            self.fit_function = partial(signal_model_wrapper2, 
                                        k2=self.parameters['k2']['value'],
                                        s_intra=s_intra)
        elif not self.parameters['k1']['optimize'] and not self.parameters['k2']['optimize'] and self.parameters['scale_factor']['optimize']:
            self.fit_function = partial(signal_model_wrapper2, 
                                        k1=self.parameters['k1']['value'],
                                        k2=self.parameters['k2']['value'],
                                        s_intra=s_intra)
        elif not self.parameters['k1']['optimize'] and self.parameters['k2']['optimize'] and self.parameters['scale_factor']['optimize']:
            self.fit_function = partial(signal_model_wrapper1,
                                        k1=self.parameters['k1']['value'],
                                        s_intra=s_intra)
    
    def get_fit(self, t, background_parameters, s_intra):
        return signal_model(t, background_parameters['k1'], background_parameters['k2'], background_parameters['scale_factor'], s_intra)
    
    def get_background(self, t, background_parameters, modulation_depth=0):
        return background_model(t, background_parameters['k1'], background_parameters['k2']) * (1-background_parameters['scale_factor']*modulation_depth)
    