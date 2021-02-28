

import numpy as np
import matplotlib.pyplot as plt


def plot_timetrace(timetrace, save_figure, directory, filename):
    plt.ioff()
    plt.plot(timetrace['t'], timetrace['s'])
    plt.xlabel(r'$\mathit{t}$ ($\mathit{\mu s}$)')
    plt.ylabel('Echo intensity (a.u.)')
    plt.tight_layout()
    if save_figure:
        filepath = directory + filename + ".png"
        plt.savefig(filepath, format='png', dpi=600)
    plt.close()