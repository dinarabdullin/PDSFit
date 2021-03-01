''' Read a libconfig array '''

import libconf

data_types = {'float': float, 'int': int, 'str': str}

def read_array(array_obj, data_type, scale=1):
    lc_array = []
    if (array_obj != []):
        for i in array_obj:
            if (data_type == 'float'):
                lc_array.append(data_types[data_type](i * scale))
            elif (data_type == 'int'):
                lc_array.append(data_types[data_type](i * scale))
            elif (data_type == 'str'):
                lc_array.append(data_types[data_type](i))      
    return lc_array