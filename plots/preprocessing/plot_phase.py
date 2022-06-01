import numpy as np
from supplement.definitions import const
import plots.set_matplotlib
import matplotlib.pyplot as plt


def plot_phase(phases, y_im_mean, y_re, y_im, y_re_new, y_im_new):
    ''' Plot phase '''
    fig = plt.figure(facecolor='w', edgecolor='w')
    
    plt.subplot(1, 3, 1)
    axes = fig.gca()
    axes.plot(phases, y_im_mean, 'k-')
    axes.set_xlim(np.amin(phases), np.amax(phases))
    axes.set_xlabel(r'Phase (deg)')
    axes.set_ylabel(r'mean(Im)')
    
    plt.subplot(1, 3, 2)
    t = np.linspace(0, y_re.size-1, y_re.size)
    axes = fig.gca()
    axes.plot(t, y_re, 'k-', label='Re')
    axes.plot(t, y_re_new, 'r-', label='Re, new phase')
    axes.set_xlim(np.amin(t), np.amax(t))
    axes.set_xlabel(r'Phase (deg)')
    axes.set_ylabel(r'Im(V(t))')
    axes.legend()
    
    plt.subplot(1, 3, 3)
    t = np.linspace(0, y_im.size-1, y_im.size)
    axes = fig.gca()
    axes.plot(t, y_im, 'k-', label='Im')
    axes.plot(t, y_im_new, 'r-', label='Im, new phase')
    axes.set_xlim(np.amin(t), np.amax(t))
    axes.set_xlabel(r'Phase (deg)')
    axes.set_ylabel(r'Im(V(t))')
    axes.legend()
    
    plt.tight_layout()
    plt.draw()
    plt.pause(0.000001)