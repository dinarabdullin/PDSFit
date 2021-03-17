import numpy as np
import plots.set_matplotlib
import matplotlib.pyplot as plt
from plots.best_layout import best_layout
from plots.plt_set_fullscreen import plt_set_fullscreen 


def plot_epr_spectrum(spectrum, save_figure=False, directory='', experiment_name=''):
    ''' Plots a simulated EPR spectrum '''
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


def plot_bandwidths_single_experiment(fig, bandwidths, experiment, spectrum=[], save_figure=False, directory=''):
    ''' 
    Plots the bandwidths of detection and pump pulses a single experiment.
    If the EPR spectrum of the spin system is provided, the bandwidths are overlayed with the EPR spectrum. 
    '''
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
        plt.pause(0.000001)
        if save_figure:
            filepath = directory + 'bandwidths_' + experiment.name + ".png"
            plt.savefig(filepath, format='png', dpi=600)


def plot_bandwidths(bandwidths, experiments, spectra=[], save_figure=False, directory=''):
    ''' 
    Plots the bandwidths of detection and pump pulses for multiple experiments .
    If the EPR spectrum of the spin system is provided, the bandwidths are overlayed with the EPR spectrum. 
    '''  
    fig = plt.figure(figsize=[10,8], facecolor='w', edgecolor='w')
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
    if save_figure:
        filepath = directory + 'bandwidths.png'
        plt.savefig(filepath, format='png', dpi=600)


def plot_time_trace(fig, simulated_time_trace, experiment, save_figure=False, directory=''):
    ''' Plots a simulated PDS time trace '''
    if fig == None:
        new_fig = plt.figure(facecolor='w', edgecolor='w')
        axes = new_fig.gca() 
    else:
        axes = fig.gca()
    axes.plot(experiment.t, experiment.s, 'k-', label="exp")
    axes.plot(simulated_time_trace['t'], simulated_time_trace['s'], 'r-', label="sim")	
    textstr = str(experiment.name)
    axes.legend(title=textstr)
    axes.set_xlabel(r'$\mathit{t}$ ($\mathit{\mu s}$)')
    axes.set_ylabel('Echo intensity (arb.u.)')
    axes.set_xlim([np.amin(experiment.t), np.amax(experiment.t)])
    axes.set_ylim([np.amin([experiment.s, simulated_time_trace['s']])-0.05, 1.05])
    if fig == None:   
        plt.tight_layout()
        plt.draw()
        if save_figure:
            filepath = directory + 'time_trace_' + experiment.name + ".png"
            plt.savefig(filepath, format='png', dpi=600)


def plot_time_traces(simulated_time_traces, experiments, save_figure=False, directory=''):
    ''' Plots simulated PDS time traces '''
    fig = plt.figure(figsize=[10,8], facecolor='w', edgecolor='w')  
    figsize = fig.get_size_inches()*fig.dpi
    num_subplots = len(experiments)
    layout = best_layout(figsize[0], figsize[1], num_subplots)
    for i in range(num_subplots):
        plt.subplot(layout[1], layout[0], i+1)
        plot_time_trace(fig, simulated_time_traces[i], experiments[i])
    plt.tight_layout() 
    plt_set_fullscreen() 
    plt.draw()
    plt.pause(0.000001)
    if save_figure:
        filepath = directory + 'time_traces.png'
        plt.savefig(filepath, format='png', dpi=600)


def plot_simulation_output(epr_spectra, bandwidths, simulated_time_traces, experiments, save_figures, directory):
    ''' Plots the simulation output '''
    # # Plot the spectrum        
    # plot_epr_spectrum(epr_spectra[0], save_figures, directory, experiments[0].name)
    # Plot the bandwidths on top of the EPR spectrum of the spin system
    plot_bandwidths(bandwidths, experiments, epr_spectra, save_figures, directory)
    # Plot the time traces
    plot_time_traces(simulated_time_traces, experiments, save_figures, directory)