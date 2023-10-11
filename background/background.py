from scipy.optimize import curve_fit


class Background:
    """PDS background."""
    
    def __init__(self):
        self.parameter_names = []
        self.parameters = {}
        self.scoring_function = None
        self.p0 = []
        self.lower_bounds = []
        self.upper_bounds = []
    
    
    def set_parameters(self, parameters):
        """Set parameters."""
        self.parameters = parameters
        self.p0 = []
        self.lower_bounds = []
        self.upper_bounds = []
        for parameter_name in self.parameter_names:
            optimize_flag = self.parameters[parameter_name]["optimize"]
            if optimize_flag:
                self.p0.append(self.parameters[parameter_name]["value"])
                self.lower_bounds.append(self.parameters[parameter_name]["range"][0])
                self.upper_bounds.append(self.parameters[parameter_name]["range"][1])
    
    
    def set_scoring_function(self, s_intra): 
        """Set the scoring function."""
    
    
    def optimize_parameters(self, t, s_exp, s_intra):
        """Optimize the background parameters to minimize MSD:
        s_exp(t) = s_inter(t, parameters) * (1 - scale_factor * s_intra(t)),
        where
        s_exp(t) is the experimental PDS time trace,
        s_intra(t) is the simulated form factor,
        s_inter(t) is the simulated background,
        parameters are the background parameters,
        scale_factor is the scale factor of the modulation depth parameter."""
        if self.lower_bounds != [] and self.upper_bounds != []:
            self.set_scoring_function(s_intra[1:])
            popt, pcov = curve_fit(
                self.scoring_function, 
                t[1:], s_exp[1:], 
                p0 = self.p0, 
                bounds = (self.lower_bounds, self.upper_bounds), 
                maxfev = 100000
                )
        background_parameters = {}
        count = 0
        for parameter_name in self.parameter_names:
            if self.parameters[parameter_name]["optimize"]:
                background_parameters[parameter_name] = popt[count]
                count += 1
            else:
                background_parameters[parameter_name] = self.parameters[parameter_name]["value"]
        return background_parameters
    
    
    def get_fit(self, t, background_parameters, s_intra):
        """Compute the fit to a PDS time trace."""
    
    
    def get_background(self, t, background_parameters, modulation_depth):
        """Compute the background."""