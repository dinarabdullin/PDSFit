class ParameterID:
    """Identifier of a fitting parameter."""

    def __init__(self, name, component):
        self.name = name
        self.component = component
        self.optimized = None
        self.index = None
        self.range = None
        self.value = None
    
    
    def set_optimized(self, opt):
        """Set the optimization flag."""
        self.optimized = opt
    
    
    def is_optimized(self):
        """Return the optimization flag."""
        return self.optimized
    
    
    def set_index(self, idx):
        """Set the index."""
        self.index = idx
    
    
    def get_index(self):
        """Return the index."""
        return self.index
    
    
    def set_range(self, ran):
        """Set the optimization range."""
        self.range = ran
    
    
    def get_range(self):
        """Return the optimization range."""
        return self.range
    
    
    def set_value(self, val):
        """Set the parameter value."""
        self.value = val

    def get_value(self):
        """Return the parameter value."""
        return self.value
    
    
    def __eq__(self, other):
        """Equality operator."""
        if isinstance(other, self.__class__):
            if self.name == other.name and self.component == other.component:
                return True
            else:
                return False
        else:
            return False
    