import numpy as np
import plots.set_matplotlib
from plots.set_matplotlib import best_rcparams
import matplotlib.pyplot as plt
from plots.best_layout import best_layout


def plot_simulated_time_trace(axes, simulated_time_trace, error_bars_simulated_time_trace, experiment):
    ''' Plots a simulated PDS time trace '''
    if error_bars_simulated_time_trace != []:
        lower_bounds = []
        upper_bounds = []
        for i in range(simulated_time_trace['s'].size):
            lower_bound = simulated_time_trace['s'][i] + error_bars_simulated_time_trace[i][0]
            upper_bound = simulated_time_trace['s'][i] + error_bars_simulated_time_trace[i][1]
            lower_bounds.append(lower_bound)
            upper_bounds.append(upper_bound)
        axes.fill_between(simulated_time_trace['t'], upper_bounds, lower_bounds, color='red', alpha=0.3, linewidth=0)
    axes.plot(experiment.t, experiment.s, 'k-', label="exp")
    axes.plot(simulated_time_trace['t'], simulated_time_trace['s'], 'r-', label="sim")	
    textstr = str(experiment.name)
    axes.legend(title=textstr)
    axes.set_xlabel(r'$\mathit{t}$ ($\mathit{\mu s}$)')
    axes.set_ylabel('Echo intensity (arb.u.)')
    axes.set_xlim([0, np.amax(experiment.t)])
    axes.set_ylim([np.amin([experiment.s, simulated_time_trace['s']])-0.05, 1.05])


def plot_simulated_time_traces(simulated_time_traces, error_bars_simulated_time_traces, experiments):
    ''' Plots simulated PDS time traces '''
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
        if error_bars_simulated_time_traces != []:
            plot_simulated_time_trace(axes, simulated_time_traces[i], error_bars_simulated_time_traces[i], experiments[i])
        else:
            plot_simulated_time_trace(axes, simulated_time_traces[i], [], experiments[i])
    plt.tight_layout() 
    return fig