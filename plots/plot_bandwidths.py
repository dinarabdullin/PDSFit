''' 
Plots the bandwidths of detection and pump pulses.
If the EPR spectrum of the spin system is provided, the bandwidths will be overlayed with the EPR spectrum 
'''

import numpy as np
import plots.set_backend
import matplotlib.pyplot as plt
import plots.set_style
from plots.best_layout import best_layout  
   
def plot_bandwidths_single_experiment(fig, bandwidths, experiment, spectrum=[], save_figure=False, directory=''):
    if fig == None:
        new_fig = plt.figure(facecolor='w', edgecolor='w')
        axes = new_fig.gca() 
    else:
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
    if fig == None:   
        plt.tight_layout()
        plt.draw()
        if save_figure:
            filepath = directory + 'bandwidths_' + experiment.name + ".png"
            plt.savefig(filepath, format='png', dpi=600)


def plot_bandwidths(bandwidths, experiments, spectra=[], save_figure=False, directory=''):
    fig = plt.figure(figsize=[10,8], facecolor='w', edgecolor='w')
    figsize = fig.get_size_inches()*fig.dpi
    num_subplots = len(experiments)
    layout = best_layout(figsize[0], figsize[1], num_subplots)
    for i in range(num_subplots):
        plt.subplot(layout[0], layout[1], i+1)
        if (spectra != []) and (len(spectra) == num_subplots):
            plot_bandwidths_single_experiment(fig, bandwidths[i], experiments[i], spectra[i])
        else:
            plot_bandwidths_single_experiment(fig, bandwidths[i], experiments[i])
    plt.tight_layout()
    plt.draw()
    if save_figure:
        filepath = directory + 'bandwidths.png'
        plt.savefig(filepath, format='png', dpi=600)