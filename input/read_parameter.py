import libconf
import sys


supported_data_types = {'float': float, 'int': int}

def read_parameter(parameter_object, parameter_name, data_type, scale=1):
    ''' 
    Reads the parameter values form a libconfig object 'parameter_object'.
    The type of the parameter values, 'data_type', can be float or int.
    Each of the parameter values is scaled by a factor 'scale'.

    The 'parameter_object' object has to have one of the following formats:
    Format (same holds for int):                                             Return:  
    parameter_object = float                                                 parameter_list = [[float]]        
    parameter_object = [float]                                               parameter_list = [[float]]   
    parameter_object = [float, float, ...]                                   parameter_list = [[float, float, ...]]
    parameter_object = (float)                                               parameter_list = [[float]]  
    parameter_object = (float, float, ...)                                   parameter_list = [[float], [float], ...]  
    parameter_object = ([float])                                             parameter_list = [[float]] 
    parameter_object = ([float], float, ... )                                parameter_list = [[float], [float], ...]                                       
    parameter_object = ([float, float, ...])                                 parameter_list = [[float, float, ...]]
    parameter_object = ([float, float, ...], float, ...)                     parameter_list = [[float, float, ...], [float], ...]
    parameter_object = ([float, float, ...], [float, float, ...], ...)       parameter_list = [[float, float, ...], [float, float, ...], ...]

    Round brakets correspond to dimension #1 of the returned list, 
    Square brakets correspond to dimension #2 of the returned list.
    '''
    parameter_list = []
    if parameter_object != () and parameter_object != []:
        if isinstance(parameter_object, tuple):
            parameter_list = []
            for component in parameter_object:
                if component != () and component != []:
                    if isinstance(component, list):
                        parameter_sublist = []
                        for subcomponent in component:
                            if isinstance(subcomponent, float) or isinstance(subcomponent, int):
                                parameter_value = (supported_data_types[data_type])(subcomponent) * (supported_data_types[data_type])(scale)
                                parameter_sublist.append(parameter_value)
                            else:
                                raise ValueError('Unsupported format of \'{0}\'!'.format(parameter_name))
                                sys.exit(1)
                    elif isinstance(component, float) or isinstance(component, int):
                        parameter_value = (supported_data_types[data_type])(component) * (supported_data_types[data_type])(scale)
                        parameter_sublist = [parameter_value]
                    else:
                        raise ValueError('Unsupported format of \'{0}\'!'.format(parameter_name))
                        sys.exit(1)
                    parameter_list.append(parameter_sublist)
                else:
                    raise ValueError('Unsupported format of \'{0}\'!'.format(parameter_name))
                    sys.exit(1)
        elif isinstance(parameter_object, list):  
            parameter_list = []
            parameter_sublist = []
            for component in parameter_object:
                if isinstance(component, float) or isinstance(component, int):
                    parameter_value = (supported_data_types[data_type])(component) * (supported_data_types[data_type])(scale)
                    parameter_sublist.append(parameter_value)
                else:
                    raise ValueError('Unsupported format of \'{0}\'!'.format(parameter_name))
                    sys.exit(1)
            parameter_list.append(parameter_sublist)
        elif isinstance(parameter_object, float) or isinstance(parameter_object, int):
            parameter_value = (supported_data_types[data_type])(parameter_object) * (supported_data_types[data_type])(scale)
            parameter_list = [[parameter_value]]
        else:
            raise ValueError('Unsupported format of \'{0}\'!'.format(parameter_name))
            sys.exit(1)
    else:
        parameter_list = [[(supported_data_types[data_type])(0)]]
    return parameter_list