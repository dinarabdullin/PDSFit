import numpy as np
import plots.set_matplotlib
from plots.set_matplotlib import best_rcparams
import matplotlib.pyplot as plt
from plots.best_layout import best_layout


def plot_simulated_time_trace(axes, simulated_time_trace, experiment):
    ''' Plots a simulated PDS time trace '''
    axes.plot(experiment.t, experiment.s, 'k-', label="exp")
    axes.plot(simulated_time_trace['t'], simulated_time_trace['s'], 'r-', label="sim")	
    textstr = str(experiment.name)
    axes.legend(title=textstr)
    axes.set_xlabel(r'$\mathit{t}$ ($\mathit{\mu s}$)')
    axes.set_ylabel('Echo intensity (arb.u.)')
    axes.set_xlim([np.amin(experiment.t), np.amax(experiment.t)])
    axes.set_ylim([np.amin([experiment.s, simulated_time_trace['s']])-0.05, 1.05])
    xl, xh = axes.get_xlim()
    yl, yh = axes.get_ylim()
    axes.set_aspect((xh-xl)/(yh-yl))


def plot_simulated_time_traces(simulated_time_traces, experiments):
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
        plot_simulated_time_trace(axes, simulated_time_traces[i], experiments[i])
    plt.tight_layout() 
    plt.draw()
    plt.pause(0.000001)
    return fig