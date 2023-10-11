import numpy as np
import plots.set_matplotlib
from plots.set_matplotlib import best_rcparams
import matplotlib.pyplot as plt
from plots.best_layout import best_layout


def plot_simulated_time_trace(axes, simulated_time_trace, experiment, error_bars = []):
    """Save an experimental PDS time trace and a corresponding simulated PDS time trace."""
    if len(error_bars) != 0:
        lower_bounds, upper_bounds = [], []
        for i in range(experiment.t.size):
            lower_bound = simulated_time_trace[i] + error_bars[i][0]
            upper_bound = simulated_time_trace[i] + error_bars[i][1]
            lower_bounds.append(lower_bound)
            upper_bounds.append(upper_bound)
        axes.fill_between(
            experiment.t, upper_bounds, lower_bounds, color = "red", alpha = 0.3, linewidth = 0
            )
    axes.plot(experiment.t, experiment.s, "k-", label="exp")
    axes.plot(experiment.t, simulated_time_trace, "r-", label="sim")	 
    axes.legend(title = str(experiment.name))
    axes.set_xlabel(r"$\mathit{t}$ ($\mathit{\mu s}$)")
    axes.set_ylabel("Echo intensity (arb.u.)")
    axes.set_xlim([0, np.amax(experiment.t)])
    axes.set_ylim([np.amin([experiment.s, simulated_time_trace]) - 0.05, 1.05])


def plot_simulated_time_traces(simulated_time_traces, experiments, error_bars = []):
    """Save experimental and simulated PDS time traces."""
    figsize = [10, 8]
    num_subplots = len(experiments)
    best_rcparams(num_subplots)
    layout = best_layout(figsize[0], figsize[1], num_subplots)
    fig = plt.figure(
        figsize = (figsize[0], figsize[1]), 
        facecolor="w", 
        edgecolor="w"
        )
    for i in range(num_subplots):
        if num_subplots == 1:
            axes = fig.gca()
        else:
            axes = fig.add_subplot(layout[0], layout[1], i+1)
        if len(error_bars) == 0:
            plot_simulated_time_trace(axes, simulated_time_traces[i], experiments[i])
        else:
            plot_simulated_time_trace(axes, simulated_time_traces[i], experiments[i], error_bars[i])
    plt.tight_layout() 
    return fig