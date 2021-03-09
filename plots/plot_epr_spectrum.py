import numpy as np
import plots.set_matplotlib
import matplotlib.pyplot as plt


def plot_epr_spectrum(spectrum, save_figure=False, directory='', experiment_name=''):
    ''' Plot a simulated EPR spectrum '''
    fig = plt.figure(facecolor='w', edgecolor='w')
    axes = fig.gca()
    axes.plot(spectrum['f'], spectrum['p']/np.amax(spectrum['p']), 'k-')
    axes.set_xlabel(r'Frequency (GHz)')
    axes.set_ylabel('Intensity (arb. u.)')
    axes.set_xlim(np.amin(spectrum['f']), np.amax(spectrum['f']))
    axes.set_ylim(0.0, 1.1)
    plt.tight_layout()
    plt.draw()
    plt.pause(0.000001) # needed for displaying the plot
    #plt.show(block=False)
    if save_figure:
        filepath = directory + 'epr_spectrum_' + experiment_name + '.png'
        plt.savefig(filepath, format='png', dpi=600)