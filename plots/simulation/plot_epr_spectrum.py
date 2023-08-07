import numpy as np
import plots.set_matplotlib
from plots.set_matplotlib import best_rcparams
import matplotlib.pyplot as plt


def plot_epr_spectrum(spectrum):
    ''' Plots a simulated EPR spectrum '''
    best_rcparams(1)
    fig = plt.figure(facecolor='w', edgecolor='w')
    axes = fig.gca()
    axes.plot(spectrum['f'], spectrum['p']/np.amax(spectrum['p']), 'k-')
    axes.set_xlabel(r'Frequency (GHz)')
    axes.set_ylabel('Intensity (arb. u.)')
    axes.set_xlim(np.amin(spectrum['f']), np.amax(spectrum['f']))
    axes.set_ylim(0.0, 1.1)
    plt.tight_layout()
    return fig