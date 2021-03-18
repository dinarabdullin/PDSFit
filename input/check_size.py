import sys


def compare_size(list1, list2, name1, name2, dim):
    if dim == 1:
        if len(list1) != len(list2):
            raise ValueError('Parameters %s and %s must have same dimensions!' % (name1, name2))
            sys.exit(1)
    elif dim == 2:
        for i in range(len(list1)):
            if len(list1[i]) != len(list2[i]):
                raise ValueError('Parameters %s and %s must have same dimensions!' % (parameter1, parameter2))
                sys.exit(1)   
    else:
        raise ValueError('Unsupported dimension of parameters %s and %s !' % (name1, name2))
        sys.exit(1)    


def nonzero_size(list1, name1, dim):
    if dim == 1:
        if len(list1) == 0:
            raise ValueError('Parameter %s must have at least one value!' % (name1))
            sys.exit(1)
    elif dim == 2:
        if len(list1[0]) == 0:
            raise ValueError('Parameter %s must have at least one value!' % (name1))
            sys.exit(1)
    else:
        raise ValueError('Unsupported dimension of parameter %s !' % (name1))
        sys.exit(1)