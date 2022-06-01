import numpy as np
from supplement.definitions import const
import plots.set_matplotlib
import matplotlib.pyplot as plt


def plot_zero_point(indices, moments, idx_zp, t, y_re, t_grid, y_grid, t_zp):
    ''' Plot zero point '''
    fig = plt.figure(facecolor='w', edgecolor='w')
    
    plt.subplot(1, 2, 1)
    axes = fig.gca()
    axes.plot(indices, moments, 'k-', label='1st moment')
    axes.axvline(x=idx_zp, color='r', linestyle='--', label='zero point')
    axes.set_xlim(np.amin(indices), np.amax(indices))
    axes.set_xlabel(r'No. data point')
    axes.set_ylabel(r'1st moment')
    
    plt.subplot(1, 2, 2)
    axes = fig.gca()
    axes.plot(t, y_re, 'k-', label='exp. time trace')
    axes.plot(t_grid, y_grid, 'g-', label='extrapolation')
    plt.axvline(x=t_zp, color='r', linestyle='--', label='zero point')
    axes.set_xlim(np.amin(t), 0.3 * np.amax(t))
    axes.set_xlabel(r'$\mathit{t}$ ($\mathit{\mu s}$)')
    axes.set_ylabel('Echo intensity (arb.u.)')
    axes.legend()
    
    plt.tight_layout()
    plt.draw()
    plt.pause(0.000001)