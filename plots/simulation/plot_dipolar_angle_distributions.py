import numpy as np
import plots.set_matplotlib
from plots.set_matplotlib import best_rcparams
import matplotlib.pyplot as plt
from plots.best_layout import best_layout


def plot_dipolar_angle_distribution(axes, dipolar_angle_distribution, experiment):
    """Plot a simulated distribution of the dipolar angle."""
    axes.plot(dipolar_angle_distribution["angle"], dipolar_angle_distribution["prob"], "k-", label = "sim")
    distr_no_selectivity = 0.5 * np.sin(np.pi / 180 * dipolar_angle_distribution["angle"])
    axes.plot(dipolar_angle_distribution["angle"], distr_no_selectivity, "r-", label = r"sin($\mathit{\theta}$)")	
    axes.legend(title = str(experiment.name))
    axes.set_xlabel(r"$\mathit{\theta}$ ($^\circ$)")
    axes.set_ylabel("Probability")  
    axes.set_xlim([0, 180])


def plot_dipolar_angle_distributions(dipolar_angle_distributions, experiments):
    """Plot simulated distributions of the dipolar angle."""
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
        plot_dipolar_angle_distribution(axes, dipolar_angle_distributions[i], experiments[i])
    plt.tight_layout() 
    return fig