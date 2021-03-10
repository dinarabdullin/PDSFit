import sys


def validate_simulation_parameters(simulation_parameters, parameter1, parameter2=''):
    ''' Validate simulation parameters '''
    if parameter2 != '':
        if len(simulation_parameters[parameter1][0]) == 0:
            raise ValueError('Parameter %s must have at least one value!' % (parameter1))
            sys.exit(1)
        if len(simulation_parameters[parameter2][0]) == 0:
            raise ValueError('Parameter %s must have at least one value!' % (parameter2))
            sys.exit(1)
        if len(simulation_parameters[parameter1]) != len(simulation_parameters[parameter2]):
            raise ValueError('Parameters %s and %s must have same dimensions!' % (parameter1, parameter2))
            sys.exit(1)   
        for i in range(len(simulation_parameters[parameter1])):
            if len(simulation_parameters[parameter1][i]) != len(simulation_parameters[parameter2][i]):
                raise ValueError('Parameters %s and %s must have same dimensions!' % (parameter1, parameter2))
                sys.exit(1)
    else:
        if len(simulation_parameters[parameter1][0]) == 0:
            raise ValueError('Parameter %s must have at least one value!' % (parameter1))
            sys.exit(1)