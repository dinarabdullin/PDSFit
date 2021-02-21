

import numpy as np
import matplotlib.pyplot as plt


def plot_timetrace(t, s, save_figure, directory, filename):
    plt.plot(t, s)
    plt.show()
    if save_figure:
        filepath = directory + filename + ".png"
        plt.savefig(filepath, format='png', dpi=600)