import numpy as np
import plots.set_matplotlib
from plots.set_matplotlib import best_rcparams
import matplotlib.pyplot as plt
from plots.best_layout import best_layout


def plot_form_factor(axes, form_factor, experiment):
    """Plot form factors for an experimental PDS time trace and 
    a corresponding simulated PDS time trace."""
    axes.plot(experiment.t, form_factor["exp"], "k-", label="exp")
    axes.plot(experiment.t, form_factor["sim"], "r-", label="sim")	
    axes.legend(title=str(experiment.name))
    axes.set_xlabel(r"$\mathit{t}$ ($\mathit{\mu s}$)")
    axes.set_ylabel("Amplitude (arb.u.)")
    axes.set_xlim([0, np.amax(experiment.t)])
    axes.set_ylim([np.amin([form_factor["exp"], form_factor["sim"]]) - 0.05, 1.05])


def plot_form_factors(form_factors, experiments):
    """Plot form factors for experimental and simulated PDS time traces."""
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
        plot_form_factor(axes, form_factors[i], experiments[i])
    plt.tight_layout() 
    return fig