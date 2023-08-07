import numpy as np
import plots.set_matplotlib
from plots.set_matplotlib import best_rcparams
import matplotlib.pyplot as plt
from plots.best_layout import best_layout


def plot_simulated_spectrum(axes, simulated_spectrum, experiment):
    ''' Plots a simulated PDS time trace and a background fit '''
    axes.plot(simulated_spectrum['f'], simulated_spectrum['pe'], 'k-', label="exp")
    axes.plot(simulated_spectrum['f'], simulated_spectrum['p'], 'r-', label="sim")	
    textstr = str(experiment.name)
    axes.legend(title=textstr)
    axes.set_xlabel(r'Frequency (MHz)')
    axes.set_ylabel('Amplitude (arb.u.)')  
    axes.set_xlim([np.amin(simulated_spectrum['f']), np.amax(simulated_spectrum['f'])])
    axes.set_ylim([np.amin(simulated_spectrum['pe'])-0.1, 1.1])


def plot_simulated_spectra(simulated_spectra, experiments):
    ''' Plots PDS time traces and background fits'''
    figsize = [10, 8]
    num_subplots = len(experiments)
    best_rcparams(num_subplots)
    layout = best_layout(figsize[0], figsize[1], num_subplots)
    fig = plt.figure(figsize=(figsize[0], figsize[1]), facecolor='w', edgecolor='w')
    for i in range(num_subplots):
        if num_subplots == 1:
            axes = fig.gca()
        else:
            axes = fig.add_subplot(layout[0], layout[1], i+1)
        plot_simulated_spectrum(axes, simulated_spectra[i], experiments[i])
    plt.tight_layout() 
    return fig