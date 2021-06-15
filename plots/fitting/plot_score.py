import numpy as np
import plots.set_matplotlib
from plots.set_matplotlib import best_rcparams
import matplotlib.pyplot as plt
from supplement.definitions import const


def plot_score(score, goodness_of_fit):
    ''' Plots the score as a function of optimization step '''
    best_rcparams(1)
    num_points = len(score)
    x = np.linspace(1,num_points,num_points)
    y = score
    fig = plt.figure(facecolor='w', edgecolor='w')
    axes = fig.gca()
    axes.semilogy(x, y, linestyle='-', marker='o', color='k')
    axes.set_xlim(0, x[-1] + 1)
    plt.xlabel('No. iteration')
    plt.ylabel(const['goodness_of_fit_axes_labels'][goodness_of_fit])
    plt.grid(True)
    plt.tight_layout()
    plt.draw()
    plt.pause(0.000001)
    return fig


def update_score_plot(fig, score, goodness_of_fit):
    ''' Re-plots the score as a function of optimization step '''
    best_rcparams(1)
    num_points = len(score)
    x = np.linspace(1,num_points,num_points)
    y = score
    axes = fig.gca()
    axes.clear()
    axes.semilogy(x, y, linestyle='-', marker='o', color='k')
    axes.set_xlim(0, x[-1] + 1)
    plt.xlabel('The number of optimization steps')
    plt.xlabel('No. iteration')
    plt.ylabel(const['goodness_of_fit_axes_labels'][goodness_of_fit])		
    plt.grid(True)
    plt.tight_layout()
    plt.draw()
    plt.pause(0.000001)


def close_score_plot(fig):
    ''' Closes the score plot '''
    plt.close(fig)