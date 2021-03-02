''' Find the best layout of multiple subplots for a given screen size'''

import math

def best_square_size(w, h, n):
    ''' 
    Given a rectangle with width and height, fill it with n squares of equal size such 
    that the squares cover as much of the rectangle's area as possible. 
    The size of a single square should be returned.
    Source:
    https://stackoverflow.com/questions/6463297/algorithm-to-fill-rectangle-with-small-squares
    '''
    hi, lo = float(max(w, h)), 0.0
    while abs(hi - lo) > 0.000001:
        mid = (lo+hi)/2.0
        midval = math.floor(w / mid) * math.floor(h / mid)
        if midval >= n:
            lo = mid
        elif midval < n: 
            hi = mid
    return min(w/math.floor(w/lo), h/math.floor(h/lo))

def best_layout(w, h, n):
    a = best_square_size(w, h, n)
    return [math.floor(w/a), math.floor(h/a)]