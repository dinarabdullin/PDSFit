import numpy as np
from functools import partial
from background.background import Background


def background_model(t, c1, c2, c3, c4):
    return 1 + c1 * np.abs(t) + c2 * np.abs(t)**2 + c3 * np.abs(t)**3 + c4 * np.abs(t)**4


def signal_model(t, c1, c2, c3, c4, scale_factor, s_intra):
    return background_model(t, c1, c2, c3, c4) * (1 + scale_factor * (s_intra - 1))


def signal_model_wrapper(t, scale_factor, c1, c2, c3, c4, s_intra):
    return signal_model(t, c1, c2, c3, c4, scale_factor, s_intra)
    

class FourthOrderPolynomialBackground(Background):
    """Background model: fourth-order polynom."""
    
    def __init__(self):
        super().__init__() 
        self.parameter_names = {
            "c1": "float", 
            "c2": "float", 
            "c3": "float", 
            "c4": "float", 
            "scale_factor": "float",
            }
        self.parameter_full_names = {
            "c1": "c1", 
            "c2": "c2", 
            "c3": "c3", 
            "c4": "c4", 
            "scale_factor": "Scale factor"
            }
    
    
    def set_scoring_function(self, s_intra):
        """Set the scoring function."""
        if self.parameters["c1"]["optimize"] and \
            self.parameters["c2"]["optimize"] and \
            self.parameters["c3"]["optimize"] and \
            self.parameters["c4"]["optimize"] and \
            self.parameters["scale_factor"]["optimize"]:
            self.scoring_function = partial(
                signal_model, 
                s_intra = s_intra
                )
        elif self.parameters["c1"]["optimize"] and \
            self.parameters["c2"]["optimize"] and \
            self.parameters["c3"]["optimize"] and \
            self.parameters["c4"]["optimize"] and \
            not self.parameters["scale_factor"]["optimize"]:
            self.scoring_function = partial(
                signal_model,
                scale_factor = self.parameters["scale_factor"]["value"],            
                s_intra = s_intra
                )                                
        elif not self.parameters["c1"]["optimize"] and \
            not self.parameters["c2"]["optimize"] and \
            not self.parameters["c3"]["optimize"] and \
            not self.parameters["c4"]["optimize"] and \
            self.parameters["scale_factor"]["optimize"]:
            self.scoring_function = partial(
                signal_model_wrapper, 
                c1 = self.parameters["c1"]["value"],
                c2 = self.parameters["c2"]["value"],
                c3 = self.parameters["c3"]["value"],
                c4 = self.parameters["c4"]["value"],
                s_intra = s_intra
                ) 
        elif not self.parameters["c1"]["optimize"] and \
            not self.parameters["c2"]["optimize"] and \
            not self.parameters["c3"]["optimize"] and \
            not self.parameters["c4"]["optimize"] and \
            not self.parameters["scale_factor"]["optimize"]:
            self.scoring_function = partial(
                signal_model,
                c1 = self.parameters["c1"]["value"],
                c2 = self.parameters["c2"]["value"],
                c3 = self.parameters["c3"]["value"],
                c4 = self.parameters["c4"]["value"],
                scale_factor = self.parameters["scale_factor"]["value"],
                s_intra = s_intra
                )                                
        else:
            raise ValueError("The polynomial coefficients can not be optimized separately!")
            sys.exit(1)
    
    
    def get_fit(self, t, background_parameters, s_intra):
        return signal_model(
            t, background_parameters["c1"], background_parameters["c2"], background_parameters["c3"], 
            background_parameters["c4"], background_parameters["scale_factor"], s_intra
            )
    
    
    def get_background(self, t, background_parameters, modulation_depth = 0):
        return background_model(t, background_parameters["c1"], background_parameters["c2"], background_parameters["c3"], background_parameters["c4"]) * \
            (1 - background_parameters["scale_factor"] * modulation_depth)