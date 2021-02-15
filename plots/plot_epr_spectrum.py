'''
Plot a simulated EPR spectrum
'''

import numpy as np
import plots.set_backend
import matplotlib.pyplot as plt
import plots.set_style


def plot_epr_spectrum(spectrum, detection_bandwidth={}, pump_bandwidth={}, save_figure=False, directory='', filename="epr_spectrum"):
    fig = plt.figure(facecolor='w', edgecolor='w')
    axes = fig.gca()
    axes.plot(spectrum["f"], spectrum["s"]/np.amax(spectrum["s"]), 'k-', label="EPR spectrum")
    if detection_bandwidth != {}:
        axes.plot(detection_bandwidth["f"], detection_bandwidth["p"]/np.amax(detection_bandwidth["p"]), 'r-', label="detection bandwidth")
    if pump_bandwidth != {}:
        axes.plot(pump_bandwidth["f"], pump_bandwidth["p"]/np.amax(pump_bandwidth["p"]), 'b-', label="pump bandwidth")
    axes.set_xlabel(r'Frequency (GHz)')
    axes.set_ylabel('Intensity (arb. u.)')
    if detection_bandwidth != {} or pump_bandwidth != {}:
        axes.legend()
    axes.set_xlim(np.amin(spectrum["f"]), np.amax(spectrum["f"]))
    axes.set_ylim(0.0, 1.1)
    plt.tight_layout()
    plt.draw()
    if save_figure:
        filepath = directory + filename + ".png"
        plt.savefig(filepath, format='png', dpi=600)