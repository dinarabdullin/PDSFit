import numpy as np
import plots.set_matplotlib
import matplotlib.pyplot as plt
from plots.best_layout import best_layout
from plots.plt_set_fullscreen import plt_set_fullscreen
from plots.plot_simulation_output import plot_time_trace


def plot_goodness_of_fit(goodness_of_fit, save_figures=False, directory=''):
    ''' Plots the goodness-of-fit as a function of optimization step '''
    num_points = len(goodness_of_fit)
    x = np.linspace(1,num_points,num_points)
    y = goodness_of_fit
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
    if save_figures:
        filepath = directory + 'goodness_of_fit.png'
        plt.savefig(filepath, format='png', dpi=600)
    return fig


def update_goodness_of_fit_plot(fig, goodness_of_fit, save_figures=False, directory=''):
    ''' Re-plots the goodness-of-fit as a function of optimization step '''
    num_points = len(goodness_of_fit)
    x = np.linspace(1,num_points,num_points)
    y = goodness_of_fit
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
    if save_figures:
        filepath = directory + 'goodness_of_fit.png'
        plt.savefig(filepath, format='png', dpi=600)


def close_goodness_of_fit_plot(fig):
    ''' Closes the goodness-of-fit plot '''
    plt.close(fig)


def plot_fits(simulated_time_traces, experiments, save_figure=False, directory=''):
    ''' Saved fits to the experimental PDS time traces '''
    fig = plt.figure(figsize=[10,8], facecolor='w', edgecolor='w')  
    figsize = fig.get_size_inches()*fig.dpi
    num_subplots = len(experiments)
    layout = best_layout(figsize[0], figsize[1], num_subplots)
    for i in range(num_subplots):
        plt.subplot(layout[1], layout[0], i+1)
        plot_time_trace(fig, simulated_time_traces[i], experiments[i])
    plt.tight_layout() 
    plt_set_fullscreen() 
    plt.draw()
    plt.pause(0.000001)
    if save_figure:
        filepath = directory + 'fits.png'
        plt.savefig(filepath, format='png', dpi=600)


def plot_fitting_output(goodness_of_fit, simulated_time_traces, experiments, save_figures, directory):
    ''' Saves the fitting output '''
    # Plot the goodness-of-fit as a function of optimization step
    fig = plot_goodness_of_fit(goodness_of_fit, save_figures, directory)
    # Plot the fits to the experimental PDS time traces
    plot_fits(simulated_time_traces, experiments, save_figures, directory)