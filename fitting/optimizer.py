class Optimizer():
    """Global optimization."""

    def __init__(self, name):
        self.name = name
        self.goodness_of_fit = None
        self.scoring_function = None
    
    
    def set_goodness_of_fit(self, goodness_of_fit):
        """Set the goodness-of-fit."""
        self.goodness_of_fit = goodness_of_fit
    
    
    def set_scoring_function(self, func):
        """Set the scoring_function."""
        self.scoring_function = func 