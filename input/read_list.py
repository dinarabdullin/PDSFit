import libconf


supported_data_types = {'float': float, 'int': int}


def read_list(list_object, data_type, scale=1):
    ''' 
    Read a libconfig list. 
    The type of the parameter values, 'data_type', can be float or int.
    Each of the parameter values is scaled by a factor 'scale'.
    '''
    lc_list = []
    if (list_object != []):
        for component in list_object:
            if (data_type == 'float'):
                lc_list.append((supported_data_types[data_type])(component) * (supported_data_types[data_type])(scale))
            elif (data_type == 'int'):
                lc_list.append((supported_data_types[data_type])(component) * (supported_data_types[data_type])(scale))
            else:
                raise ValueError('Unsupported format!')
                sys.exit(1) 
    return lc_list