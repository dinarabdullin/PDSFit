import numpy as np
import plots.set_matplotlib
from plots.set_matplotlib import best_rcparams
import matplotlib.pyplot as plt
from plots.best_layout import best_layout


def plot_background_free_time_trace(axes, background_free_time_trace, experiment):
    ''' Plots a simulated PDS time trace and a background fit '''
    axes.plot(background_free_time_trace['t'], background_free_time_trace['se'], 'k-', label="exp")
    axes.plot(background_free_time_trace['t'], background_free_time_trace['s'], 'r-', label="sim")	
    textstr = str(experiment.name)
    axes.legend(title=textstr)
    axes.set_xlabel(r'$\mathit{t}$ ($\mathit{\mu s}$)')
    axes.set_ylabel('Amplitude (arb.u.)')
    axes.set_xlim([0, np.amax(experiment.t)])
    axes.set_ylim([np.amin([background_free_time_trace['se'], background_free_time_trace['s']])-0.05, 1.05])


def plot_background_free_time_traces(background_free_time_traces, experiments):
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
        plot_background_free_time_trace(axes, background_free_time_traces[i], experiments[i])
    plt.tight_layout() 
    plt.draw()
    plt.pause(0.000001)
    return fig