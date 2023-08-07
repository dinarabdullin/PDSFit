import numpy as np
import plots.set_matplotlib
from plots.set_matplotlib import best_rcparams
import matplotlib.pyplot as plt
from plots.best_layout import best_layout


def plot_background_time_trace(axes, background_time_trace, error_bars_background_time_trace, experiment):
    ''' Plots a simulated PDS time trace and a background fit '''
    if error_bars_background_time_trace != []:
        lower_bounds = []
        upper_bounds = []
        for i in range(background_time_trace['s'].size):
            lower_bound = background_time_trace['s'][i] + error_bars_background_time_trace[i][0]
            upper_bound = background_time_trace['s'][i] + error_bars_background_time_trace[i][1]
            lower_bounds.append(lower_bound)
            upper_bounds.append(upper_bound)
        axes.fill_between(background_time_trace['t'], lower_bounds, upper_bounds, color='red', alpha=0.3, linewidth=0.0)
    axes.plot(experiment.t, experiment.s, 'k-', label="exp")
    axes.plot(background_time_trace['t'], background_time_trace['s'], 'r-', label="bckg")	
    textstr = str(experiment.name)
    axes.legend(title=textstr)
    axes.set_xlabel(r'$\mathit{t}$ ($\mathit{\mu s}$)')
    axes.set_ylabel('Echo intensity (arb.u.)')
    axes.set_xlim([0, np.amax(experiment.t)])
    axes.set_ylim([np.amin([experiment.s, background_time_trace['s']])-0.05, 1.05])


def plot_background_time_traces(background_time_traces, error_bars_background_time_traces, experiments):
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
        if error_bars_background_time_traces != []:
            plot_background_time_trace(axes, background_time_traces[i], error_bars_background_time_traces[i], experiments[i])
        else:
            plot_background_time_trace(axes, background_time_traces[i], [], experiments[i])
    plt.tight_layout() 
    return fig