import sys


def libconf2data(container, data_type = "default"):
    """Read out the data from the libconf container as X or List[X],
    where X = str, int, bool, or float"""
    if isinstance(container, tuple):
        data = []
        for item in container:
            if isinstance(item, list):
                row = []
                for element in item:
                    row.append(change_data_type(element, data_type))
                data.append(row)
            else:
                data.append(item)
    elif isinstance(container, list):
        data = []
        for item in container:
            data.append(change_data_type(item, data_type))
    else:
        data = change_data_type(container, data_type)
    return data


def change_data_type(x, data_type):
    """Convert str to X, where X = int, bool, or float."""
    if data_type == "default" or data_type == "str":
        return x
    elif data_type == "float":
        return float(x)
    elif data_type == "int":
        return int(x)
    elif data_type == "bool":
        return bool(x)
    else:
        raise ValueError("Unsupported format!")
        sys.exit(1)