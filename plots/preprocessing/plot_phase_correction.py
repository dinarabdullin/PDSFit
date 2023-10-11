import numpy as np
import plots.set_matplotlib
import matplotlib.pyplot as plt


def plot_y_imag_vs_phase(phases, y_im_mean):
    """Plot the mean value of the imaginary part of the PDS time trace vs phase.""" 
    fig = plt.figure(facecolor = "w", edgecolor = "w")
    axes = fig.gca()
    axes.plot(phases, y_im_mean, "k-")
    axes.set_xlim(np.amin(phases), np.amax(phases))
    axes.set_xlabel(r"Phase (deg)")
    axes.set_ylabel(r"mean(Im)")
    plt.tight_layout()
    plt.draw()
    plt.pause(1e-6)
    

def plot_phase_correction(y_re, y_im, y_re_corr, y_im_corr, phase):
    """Plot the PDS time trace before and after phase correction."""
    fig = plt.figure(facecolor = "w", edgecolor = "w")
    plt.subplot(1, 2, 1)
    t = np.linspace(0, y_re.size - 1, y_re.size)
    axes = fig.gca()
    axes.plot(t, y_re, "k-", label = "Re")
    axes.plot(t, y_re_new, "r-", label = "Re, new phase = {0}".format(int(phase)))
    axes.set_xlim(np.amin(t), np.amax(t))
    axes.set_xlabel(r"Phase (deg)")
    axes.set_ylabel(r"Re")
    axes.legend()
    plt.subplot(1, 2, 2)
    t = np.linspace(0, y_im.size - 1, y_im.size)
    axes = fig.gca()
    axes.plot(t, y_im, "k-", label = "Im")
    axes.plot(t, y_im_new, "r-", label = "Im, new phase = {0}".format(int(phase)))
    axes.set_xlim(np.amin(t), np.amax(t))
    axes.set_xlabel(r"Phase (deg)")
    axes.set_ylabel(r"Im")
    axes.legend()
    plt.tight_layout()
    plt.draw()
    plt.pause(1e-6)