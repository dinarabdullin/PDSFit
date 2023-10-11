import numpy as np
import plots.set_matplotlib
from plots.set_matplotlib import best_rcparams
import matplotlib.pyplot as plt
from plots.best_layout import best_layout


def plot_background(axes, background, experiment, error_bars = []):
    """Plot an experimental PDS time trace and its simulated background."""
    if len(error_bars) != 0:
        lower_bounds, upper_bounds = [], []
        for i in range(experiment.t.size):
            lower_bound = background[i] + error_bars[0][i]
            upper_bound = background[i] + error_bars[1][i]
            lower_bounds.append(lower_bound)
            upper_bounds.append(upper_bound)
        axes.fill_between(experiment.t, lower_bounds, upper_bounds, color = "red", alpha = 0.3, linewidth = 0.0)
    axes.plot(experiment.t, experiment.s, "k-", label = "exp")
    axes.plot(experiment.t, background, "r-", label = "bckg")	
    axes.legend(title = str(experiment.name))
    axes.set_xlabel(r"$\mathit{t}$ ($\mathit{\mu s}$)")
    axes.set_ylabel("Echo intensity (arb.u.)")
    axes.set_xlim([0, np.amax(experiment.t)])
    axes.set_ylim([np.amin([experiment.s, background])-0.05, 1.05])


def plot_backgrounds(backgrounds, experiments, error_bars = []):
    """Plot PDS time traces and their simulated backgrounds."""
    figsize = [10, 8]
    num_subplots = len(experiments)
    best_rcparams(num_subplots)
    layout = best_layout(figsize[0], figsize[1], num_subplots)
    fig = plt.figure(
        figsize = (figsize[0], figsize[1]), 
        facecolor = "w", 
        edgecolor = "w"
        )
    for i in range(num_subplots):
        if num_subplots == 1:
            axes = fig.gca()
        else:
            axes = fig.add_subplot(layout[0], layout[1], i+1)
        if len(error_bars) == 0:
            plot_background(axes, backgrounds[i], experiments[i])
        else:
            plot_background(axes, backgrounds[i], experiments[i], error_bars[i])
    plt.tight_layout() 
    return fig