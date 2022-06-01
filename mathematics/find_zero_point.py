import numpy as np
from scipy.interpolate import griddata
from mathematics.find_nearest import find_nearest


def find_zero_point(t, y_re):
    ''' Finds the zero point of the PDS time trace '''
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
    indices = np.arange(index_max-index_increment, index_max+index_increment+1, 1)
    moments = []
    moment_min = 0.0
    idx_zp = index_max
    for k in range(index_max-index_increment, index_max+index_increment+1, 1):
        moment = 0
        for l in range(-index_increment, index_increment+1, 1):
            moment += y_grid[k+l] * float(l)
        moments.append(moment)
        if (k == index_max-index_increment) or (np.absolute(moment) < moment_min):
            moment_min = np.absolute(moment)
            idx_zp = k
    t_zp = t_grid[idx_zp]
    #from plots.preprocessing.plot_zero_point import plot_zero_point
    #plot_zero_point(indices, moments, idx_zp, t, y_re, t_grid, y_grid, t_zp)
    return t_zp


def set_zero_point(t, y_re, y_im, t_zp):
    ''' Sets the zero point of the PDS time trace to t_zp '''
    # Make a linear time grid with a step of 1 ns and extrapolate the PDS time trace on this grid
    t_step = 0.001
    t_grid = np.arange(np.amin(t), np.amax(t), t_step)
    y_grid = griddata(t, y_re, t_grid, method='linear')
    # Determine the amplitude of the PDS signal at t_zp
    idx_zp = find_nearest(t_grid, t_zp)
    y_re_zp = y_grid[idx_zp]
    # Remove all time points before the zero point 
    idx_first_value = min(range(t.size), key=lambda i: abs(t[i]-t_zp))
    if t[idx_first_value] < t_zp:
        idx_first_value += 1
    t_new = t - t_zp * np.ones(t.size)
    t_new = t_new[idx_first_value:-1]
    y_re_new = y_re[idx_first_value:-1] / y_re_zp
    y_im_new = y_im[idx_first_value:-1] / y_re_zp
    return t_new, y_re_new, y_im_new