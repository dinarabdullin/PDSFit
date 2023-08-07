import sys


def read_list(list_object, object_name, data_type, scale=1):
    ''' 
    Reads out 'list_object'. 
    The data type of each element, 'data_type', can be either 'float' or 'int'.
    Each element is scaled by 'scale'.
    '''
    list_values = []
    if isinstance(list_object, list):
        for element in list_object:
            try:
                element_value = (data_type)(element)
                if data_type == float:
                    element_value *= (data_type)(scale)
                list_values.append(element_value)
            except ValueError:
                raise ValueError('Unsupported format of \'{0}\'!'.format(object_name))
                sys.exit(1)
    else:
        try:
            list_value = (data_type)(list_object)
            if data_type == float:
                list_value *= (data_type)(scale)
            list_values.append(list_value)
        except ValueError:
            raise ValueError('Unsupported format of \'{0}\'!'.format(object_name))
            sys.exit(1) 
    return list_values