import numpy as np
from abc import ABC, abstractmethod


class IntegrationGrid(ABC):
        
    @abstractmethod
    def get_weights(self):
        pass

    @abstractmethod
    def get_points(self):
        pass
    
    @abstractmethod
    def get_weighted_summands(self):
        pass
    
    @abstractmethod
    def integrate_function(self, function):
        pass