''' Read a libconfig list '''

import libconf
from input.read_array import read_array

data_types = {'float': float, 'int': int, 'str': str}

def read_list(list_obj, data_type, scale=1):
    if isinstance(data_type, str):
        data_type = (data_type,)
    lc_list = []
    if (list_obj != ()):
        for i in list_obj:
            if (data_type[0] == 'float'):
                lc_list.append(data_types[data_type[0]](i * scale))
            elif (data_type[0] == 'int'):
                lc_list.append(data_types[data_type[0]](i * scale))
            elif (data_type[0] == 'str'):
                lc_list.append(data_types[data_type[0]](i))
            elif (data_type[0] == 'array'):
                lc_array = read_array(i, data_type[1], scale)
                lc_list.append(lc_array)       
    return lc_list