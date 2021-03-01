''' Plot a simulated PDS time trace '''

import numpy as np
import plots.set_backend
import matplotlib.pyplot as plt
import plots.set_style

def plot_time_trace(t, s_sim, s_exp, save_figure=False, directory='', filename="timetrace"):
    fig = plt.figure(facecolor='w', edgecolor='w')
    axes = fig.gca()    
    axes.plot(t, s_exp, 'k-', label="exp")
    axes.plot(t, s_sim, 'r--', label="sim")	
    axes.legend(loc='upper right', frameon=False)
    plt.xlim([min(t), max(t)])
    plt.ylim([np.amin(s_sim)-0.1, 1.1])
    plt.xlabel(r'$\mathit{t}$ ($\mathit{\mu s}$)')
    plt.ylabel('Echo intensity (arb.u.)')
    plt.tight_layout()
    plt.draw()
    if save_figure:
        filepath = directory + filename + ".png"
        plt.savefig(filepath, format='png', dpi=600)