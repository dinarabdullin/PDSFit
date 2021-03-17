import libconf
from input.read_list import read_list


supported_data_types = {'float': float, 'int': int}


def read_tuple(tuple_object, data_type, scale=1):
    ''' 
    Read a libconfig tuple.
    The type of the parameter values, 'data_type', can be float or int.
    Each of the parameter values is scaled by a factor 'scale'.
    '''    
    lc_tuple = []
    if tuple_object != ():
        for component in tuple_object:
            if data_type[0] == 'float':
                lc_tuple.append(supported_data_types[data_type[0]](component) * supported_data_types[data_type[0]](scale))
            elif data_type[0] == 'int':
                lc_tuple.append(supported_data_types[data_type[0]](component) * supported_data_types[data_type[0]](scale))
            elif data_type[0] == 'array':
                lc_list = read_list(component, data_type[1], scale)
                lc_tuple.append(lc_list)
            else:
                raise ValueError('Unsupported format!')
                sys.exit(1)
    return lc_tuple