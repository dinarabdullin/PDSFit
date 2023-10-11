import sys
import os
import numpy as np
import plots.set_matplotlib
from plots.set_matplotlib import best_rcparams
import matplotlib.pyplot as plt


def plot_numerical_error(x, mean, std):
    fig = plt.figure(facecolor = "w", edgecolor = "w")
    n, bins, patches = plt.hist(x, 50, density = 1, color = "green", alpha = 0.7)
    probs = np.exp(-0.5 * ((bins - mean) / std)**2) / (np.sqrt(2 * np.pi) * std)
    plt.plot(bins, probs, "--", color ="black")
    plt.xlabel(r"$\mathit{\chi^2}$")
    plt.ylabel("Probability")
    plt.tight_layout()
    filepath = os.getcwd()+ "/numerical_error.png"
    fig.savefig(filepath, format = "png", dpi = 300)