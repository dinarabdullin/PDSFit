import numpy as np
import plots.set_matplotlib
from plots.set_matplotlib import best_rcparams
import matplotlib.pyplot as plt
from plots.best_layout import best_layout


def plot_simulated_spectrum(axes, dipolar_angle_distribution, experiment):
    ''' Plots a simulated PDS time trace and a background fit '''
    axes.plot(dipolar_angle_distribution['v'], dipolar_angle_distribution['p'], 'k-', label="sim")
    axes.plot(dipolar_angle_distribution['v'], 0.5*np.sin(np.pi/180*dipolar_angle_distribution['v']), 'r-', label=r'sin($\mathit{\theta}$)')	
    textstr = str(experiment.name)
    axes.legend(title=textstr)
    axes.set_xlabel(r'$\mathit{\theta}$ ($^\circ$)')
    axes.set_ylabel('Probability')  
    axes.set_xlim([0, 180])


def plot_dipolar_angles(dipolar_angle_distributions, experiments):
    ''' Plots PDS time traces and background fits'''
    figsize = [10, 8]
    num_subplots = len(experiments)
    best_rcparams(num_subplots)
    layout = best_layout(figsize[0], figsize[1], num_subplots)
    fig = plt.figure(figsize=(figsize[0], figsize[1]), facecolor='w', edgecolor='w')
    for i in range(num_subplots):
        if num_subplots == 1:
            axes = fig.gca()
        else:
            axes = fig.add_subplot(layout[0], layout[1], i+1)
        plot_simulated_spectrum(axes, dipolar_angle_distributions[i], experiments[i])
    plt.tight_layout() 
    return fig