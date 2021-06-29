import numpy as np
import plots.set_matplotlib
from plots.set_matplotlib import best_rcparams
import matplotlib
import matplotlib.pyplot as plt
from plots.best_layout import best_layout


def plot_bandwidths_single_experiment(axes, bandwidths, experiment, spectrum=[]):
    ''' 
    Plots the bandwidths of detection and pump pulses a single experiment.
    If the EPR spectrum of the spin system is provided, the bandwidths are overlayed with the EPR spectrum. 
    '''
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
    axes.set_xlabel(r'Frequency (GHz)')
    axes.set_ylabel('Intensity (arb. u.)')
    axes.set_xlim(min(f_min), max(f_max))
    axes.set_ylim(0.0, 1.1)
    textstr = str(experiment.name) + ', ' + str(experiment.magnetic_field) + ' T'
    axes.set_title(textstr, fontsize=matplotlib.rcParams['font.size'])
    # Make axes square
    xl, xh = axes.get_xlim()
    yl, yh = axes.get_ylim()
    axes.set_aspect((xh-xl)/(yh-yl))


def plot_bandwidths(bandwidths, experiments, spectra=[]):
    ''' 
    Plots the bandwidths of detection and pump pulses for multiple experiments .
    If the EPR spectrum of the spin system is provided, the bandwidths are overlayed with the EPR spectrum. 
    '''  
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
        if (spectra != []) and (len(spectra) == num_subplots):
            plot_bandwidths_single_experiment(axes, bandwidths[i], experiments[i], spectra[i])
        else:
            plot_bandwidths_single_experiment(axes, bandwidths[i], experiments[i])
    left = 0
    right = float(layout[1])/float(layout[1]+1)
    bottom = 0.5 * (1-right)
    top = 1 - bottom
    fig.tight_layout(rect=[left, bottom, right, top]) 
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(right+0.01, 0.5), frameon=False)
    plt.draw()
    plt.pause(0.000001)
    return fig