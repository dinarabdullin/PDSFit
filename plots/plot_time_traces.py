''' Plot simulated PDS time traces '''

import numpy as np
import plots.set_backend
import matplotlib.pyplot as plt
import plots.set_style
from plots.best_layout import best_layout 

def plot_time_trace(fig, simulated_time_trace, experiment, save_figure=False, directory=''):
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
    axes.set_ylim([np.amin([experiment.s, simulated_time_trace['s']])-0.1, 1.1])
    if fig == None:   
        plt.tight_layout()
        plt.draw()
        if save_figure:
            filepath = directory + 'time_trace_' + experiment.name + ".png"
            plt.savefig(filepath, format='png', dpi=600)


def plot_time_traces(simulated_time_traces, experiments, save_figure=False, directory=''):
    fig = plt.figure(figsize=[10,8], facecolor='w', edgecolor='w')
    figsize = fig.get_size_inches()*fig.dpi
    num_subplots = len(experiments)
    layout = best_layout(figsize[0], figsize[1], num_subplots)
    for i in range(num_subplots):
        plt.subplot(layout[0], layout[1], i+1)
        plot_time_trace(fig, simulated_time_traces[i], experiments[i])
    plt.tight_layout()
    plt.draw()
    if save_figure:
        filepath = directory + 'time_traces.png'
        plt.savefig(filepath, format='png', dpi=600)