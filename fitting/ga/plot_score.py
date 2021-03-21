import numpy as np
import plots.set_matplotlib
import matplotlib.pyplot as plt


def plot_score(score):
    ''' Plots the score as a function of optimization step '''
    num_points = len(score)
    x = np.linspace(1,num_points,num_points)
    y = score
    fig = plt.figure(facecolor='w', edgecolor='w')
    axes = fig.gca()
    axes.semilogy(x, y, linestyle='-', marker='o', color='k')
    axes.set_xlim(0, x[-1] + 1)
    plt.xlabel('No. iteration')
    plt.ylabel(r'$\mathit{\chi^2}$')
    plt.grid(True)
    plt.tight_layout()
    plt.draw()
    plt.pause(0.000001)
    return fig


def update_score_plot(fig, score):
    ''' Re-plots the score as a function of optimization step '''
    num_points = len(score)
    x = np.linspace(1,num_points,num_points)
    y = score
    axes = fig.gca()
    axes.clear()
    axes.semilogy(x, y, linestyle='-', marker='o', color='k')
    axes.set_xlim(0, x[-1] + 1)
    plt.xlabel('The number of optimization steps')
    plt.xlabel('No. iteration')
    plt.ylabel(r'$\mathit{\chi^2}$')		
    plt.grid(True)
    plt.tight_layout()
    plt.draw()
    plt.pause(0.000001)


def close_score_plot(fig):
    ''' Closes the score plot '''
    plt.close(fig)