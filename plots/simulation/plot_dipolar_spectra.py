import numpy as np
import plots.set_matplotlib
from plots.set_matplotlib import best_rcparams
import matplotlib.pyplot as plt
from plots.best_layout import best_layout


def plot_dipolar_spectrum(axes, dipolar_spectrum, experiment):
    """Plot a dipolar spectrum for an experimental PDS time trace 
    and a corresponding simulated PDS time trace."""
    axes.plot(dipolar_spectrum["freq"], dipolar_spectrum["exp"], "k-", label="exp")
    axes.plot(dipolar_spectrum["freq"], dipolar_spectrum["sim"], "r-", label="sim")	
    axes.legend(title = str(experiment.name))
    axes.set_xlabel(r"Frequency (MHz)")
    axes.set_ylabel("Amplitude (arb.u.)")  
    axes.set_xlim([np.amin(dipolar_spectrum["freq"]), np.amax(dipolar_spectrum["freq"])])
    axes.set_ylim([np.amin(dipolar_spectrum["exp"]) - 0.1, 1.1])


def plot_dipolar_spectra(dipolar_spectra, experiments):
    """Plot dipolar spectra for experimental and simulated PDS time traces."""
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
        plot_dipolar_spectrum(axes, dipolar_spectra[i], experiments[i])
    plt.tight_layout() 
    return fig