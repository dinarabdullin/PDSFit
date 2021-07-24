import numpy as np
from scipy.interpolate import griddata


def set_zero_point(t, y_re, y_im):
    ''' Sets the zero point of the PDS time trace '''
    # Make a linear time grid with a step of 1 ns and extrapolate the PDS time trace on this grid
    t_step = 0.001
    t_grid = np.arange(np.amin(t), np.amax(t), t_step)
    y_grid = griddata(t, y_re, t_grid, method='linear')
    # Find the maximum of the PDS time trace
    index_max = np.argmax(y_grid)
    if type(index_max) is np.ndarray:
        index_max = index_max[-1]
    # Find the minima on both sides of the maximum
    y_grad = np.diff(y_grid) / t_step
    index_lower = 0
    index_upper = t_grid.size - 1
    for i in range(index_max, index_lower, -1):
        if y_grad[i-1] <= 0:
            index_lower = i
            break
    for i in range(index_max, index_upper, 1):
        if y_grad[i] >= 0:
            index_upper = i
            break
    # Find the zero point 
    length = min([index_max-index_lower, index_upper-index_max])
    index_increment = length // 2
    if index_increment >= 2:
        c = 0
        s_min = 0.0
        for k in range(index_max-index_increment, index_max+index_increment+1, 1):
            s = 0
            for l in range(-index_increment, index_increment+1, 1):
                s += y_grid[k+l] * float(l)
            if (c == 0) or (np.absolute(s) < s_min):
                s_min = np.absolute(s)
                idx_zp = k
            c += 1
    else:
        idx_zp = index_max
    t_zp = t_grid[idx_zp]
    y_re_zp = y_grid[idx_zp]
    # Remove all time points below the zero point 
    idx_first_value = min(range(t.size), key=lambda i: abs(t[i]-t_zp))
    if t[idx_first_value] < t_zp:
        idx_first_value += 1
    t_new = t - t_zp * np.ones(t.size)
    t_new = t_new[idx_first_value:-1]
    y_re_new = y_re[idx_first_value:-1] / y_re_zp
    y_im_new = y_im[idx_first_value:-1] / y_re_zp
    return t_zp, t_new, y_re_new, y_im_new