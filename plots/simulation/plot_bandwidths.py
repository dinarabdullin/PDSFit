import numpy as np
import plots.set_matplotlib
from plots.set_matplotlib import best_rcparams
import matplotlib
import matplotlib.pyplot as plt
from plots.best_layout import best_layout


def plot_bandwidths_single_experiment(axes, bandwidths, epr_spectrum, experiment):
    """Plots the bandwidths of detection and pump pulses for a single experiment."""
    # Frequency range
    f_min, f_max = [], []
    f_min.append(np.amin(epr_spectrum["freq"]))
    f_max.append(np.amax(epr_spectrum["freq"]))
    for key in bandwidths:
        f_min.append(np.amin(bandwidths[key]["freq"]))
        f_max.append(np.amax(bandwidths[key]["freq"]))  
    # EPR spectrum
    axes.plot(epr_spectrum["freq"], epr_spectrum["prob"] / np.amax(epr_spectrum["prob"]), "k-", label = "spc")
    # Pulses' profiles
    for key in bandwidths:
        if key == "detection_bandwidth":
            axes.plot(bandwidths[key]["freq"], bandwidths[key]["prob"] / np.amax(bandwidths[key]["prob"]), "r-", label="detect")
        elif key == "pump_bandwidth":
            axes.plot(bandwidths[key]["freq"], bandwidths[key]["prob"] / np.amax(bandwidths[key]["prob"]), "b-", label="pump")  
    title = str(experiment.name) + ", " + str(experiment.magnetic_field) + " T"
    axes.set_title(label=title, fontsize = matplotlib.rcParams["font.size"])
    axes.set_xlabel(r"Frequency (GHz)")
    axes.set_ylabel("Intensity (arb. u.)")
    axes.set_xlim(min(f_min), max(f_max))
    axes.set_ylim(0.0, 1.1)
    axes.legend()


def plot_bandwidths(bandwidths, epr_spectra, experiments):
    """Plot the bandwidths of detection and pump pulses for several PDS experiments."""
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
        plot_bandwidths_single_experiment(axes, bandwidths[i], epr_spectra[i], experiments[i])
    fig.tight_layout()
    return fig