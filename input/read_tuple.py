import sys
from input.read_list import read_list


def read_tuple(tuple_object, object_name, data_type, scale=1):
    ''' 
    Reads out 'tuple_object'. 
    The data type of each element, 'data_type', can be either 'float' or 'int'.
    Each element is scaled by 'scale'.
    '''    
    tuple_values = []
    if isinstance(tuple_object, tuple):
        for element in tuple_object:
            if isinstance(element, list):
                list_values = read_list(element, object_name, data_type, scale)
                tuple_values.append(list_values)
            else:
                try:
                    element_value = (data_type)(element)
                    if data_type == float:
                        element_value *= (data_type)(scale)
                    tuple_values.append(element_value)
                except ValueError:
                    raise ValueError('Unsupported format of \'{0}\'!'.format(object_name))
                    sys.exit(1)
    elif isinstance(tuple_object, list):
        list_values = read_list(element, object_name, data_type, scale)
        tuple_values.append(list_values)
    else:
        try:
            tuple_value = (data_type)(tuple_object)
            if data_type == float:
                tuple_value *= (data_type)(scale)
            tuple_values.append(tuple_value)
        except ValueError:
            raise ValueError('Unsupported format of \'{0}\'!'.format(object_name))
            sys.exit(1) 
    return tuple_values