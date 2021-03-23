import numpy as np
import plots.set_matplotlib
import matplotlib.pyplot as plt
from plots.best_layout import best_layout
from plots.plt_set_fullscreen import plt_set_fullscreen 


def plot_bandwidths_single_experiment(fig, bandwidths, experiment, spectrum=[]):
    ''' 
    Plots the bandwidths of detection and pump pulses a single experiment.
    If the EPR spectrum of the spin system is provided, the bandwidths are overlayed with the EPR spectrum. 
    '''
    axes = fig.gca() 
    f_min, f_max = [], []
    if spectrum != []:
        axes.plot(spectrum['f'], spectrum['p']/np.amax(spectrum['p']), 'k-', label='EPR spectrum')
        f_min.append(np.amin(spectrum['f']))
        f_max.append(np.amax(spectrum['f']))
    for key in bandwidths:
        if key == 'detection_bandwidth':
            axes.plot(bandwidths[key]['f'], bandwidths[key]['p']/np.amax(bandwidths[key]['p']), 'r-', label='detection bandwidth')
        elif key == 'pump_bandwidth':
            axes.plot(bandwidths[key]['f'], bandwidths[key]['p']/np.amax(bandwidths[key]['p']), 'b-', label='pump bandwidth')
        f_min.append(np.amin(bandwidths[key]['f']))
        f_max.append(np.amax(bandwidths[key]['f']))  
    textstr = str(experiment.name) + ', ' + str(experiment.magnetic_field) + ' T'
    axes.legend(title=textstr)
    axes.set_xlabel(r'Frequency (GHz)')
    axes.set_ylabel('Intensity (arb. u.)')
    axes.set_xlim(min(f_min), max(f_max))
    axes.set_ylim(0.0, 1.1)


def plot_bandwidths(bandwidths, experiments, spectra=[]):
    ''' 
    Plots the bandwidths of detection and pump pulses for multiple experiments .
    If the EPR spectrum of the spin system is provided, the bandwidths are overlayed with the EPR spectrum. 
    '''  
    fig = plt.figure(figsize=(10,8), facecolor='w', edgecolor='w')
    figsize = fig.get_size_inches()*fig.dpi
    num_subplots = len(experiments)
    layout = best_layout(figsize[0], figsize[1], num_subplots)
    for i in range(num_subplots):
        plt.subplot(layout[1], layout[0], i+1)
        if (spectra != []) and (len(spectra) == num_subplots):
            plot_bandwidths_single_experiment(fig, bandwidths[i], experiments[i], spectra[i])
        else:
            plot_bandwidths_single_experiment(fig, bandwidths[i], experiments[i])
    plt.tight_layout()
    plt_set_fullscreen()
    plt.draw()
    plt.pause(0.000001)
    return fig