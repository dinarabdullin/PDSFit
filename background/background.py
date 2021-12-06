from scipy.optimize import curve_fit


class Background:
    ''' Background class '''
    
    def __init__(self):
        self.parameter_names = []
        self.parameters = {}
        self.fit_function = None
        self.p0 = []
        self.lower_bounds = []
        self.upper_bounds = []
    
    def set_parameters(self, parameters):
        ''' Set the parameters '''
        self.parameters = parameters
        self.p0 = []
        self.lower_bounds = []
        self.upper_bounds = []
        for parameter_name in self.parameter_names:
            optimize_flag = self.parameters[parameter_name]['optimize']
            if optimize_flag:
                self.p0.append(self.parameters[parameter_name]['value'])
                self.lower_bounds.append(self.parameters[parameter_name]['range'][0])
                self.upper_bounds.append(self.parameters[parameter_name]['range'][1])
    
    def set_fit_function(self, s_intra): 
        ''' Set the fit function '''
    
    def optimize_parameters(self, t, s_exp, s_intra):
        ''' 
        Optimize the parameters to yeild the best fit to:
        s_exp(t) = s_inter(t, parameters) * (1 - scale_factor * s_intra(t)),
        where
        s_exp(t) - experimental time trace
        s_intra(t) - simulated intramolecular part of the time trace (that is modulated by dipolar frequencies),,
        s_inter(t) - simulated intermolecular part of the time trace (background),
        parameters - background parameters,
        scale_factor - scale factor of the modulation depth parameter (included into the background parameters).
        '''
        if self.lower_bounds != [] and self.upper_bounds != []:
            self.set_fit_function(s_intra[1:])
            popt, pcov = curve_fit(self.fit_function, t[1:], s_exp[1:], p0=self.p0, bounds=(self.lower_bounds, self.upper_bounds), maxfev=100000)
        background_parameters = {}
        count = 0
        for parameter_name in self.parameter_names:
            if self.parameters[parameter_name]['optimize']:
                background_parameters[parameter_name] = popt[count]
                count += 1
            else:
                background_parameters[parameter_name] = self.parameters[parameter_name]['value']
        return background_parameters
        
    def get_fit(self, t, background_parameters, s_intra):
        ''' Compute the fit to the PDS time trace '''
    
    def get_background(self, t, background_parameters, modulation_depth):
        ''' Compute the background fit '''