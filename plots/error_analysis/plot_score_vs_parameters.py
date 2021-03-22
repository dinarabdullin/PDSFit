import sys
import numpy as np
from scipy.interpolate import griddata
import plots.set_matplotlib
import matplotlib.pyplot as plt
from plots.best_layout import best_layout
from plots.plt_set_fullscreen import plt_set_fullscreen 
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
from supplement.definitions import const    


def plot_1d(fig, error_analysis_parameters, score_vs_parameter_set, optimized_parameters, fitting_parameters, score_threshold):
    # Set the fitting parameter
    parameter_id = error_analysis_parameters[0]
    # Set the values of the fitting parameter and the corresponding score values
    x = score_vs_parameter_set['parameters'][0] / const['fitting_parameters_scales'][parameter_id.name]
    y = score_vs_parameter_set['score']
    # Set the color code for y
    cmin = np.amin(y) + score_threshold
    cmax = 2 * cmin
    # Plot the figure
    axes = fig.gca()
    im = axes.scatter(x, y, c=y, cmap='jet_r', vmin=cmin, vmax=cmax)
    axes.set_xlim(round(np.amin(x),1), round(np.amax(x),1))
    xlabel_text = const['fitting_parameters_labels'][parameter_id.name][0] + \
                  r'$_{%d, %d}$' % (parameter_id.spin_pair+1, parameter_id.component+1) + ' ' + \
                  const['fitting_parameters_labels'][parameter_id.name][1]
    ylabel_text = r'$\mathit{\chi^2}$'
    axes.set_xlabel(xlabel_text)
    axes.set_ylabel(ylabel_text)
    # Depict the optimized fitting parameter
    parameter_index = parameter_id.get_index(fitting_parameters['indices']) 
    x_opt = optimized_parameters[parameter_index] / const['fitting_parameters_scales'][parameter_id.name]
    y_opt = np.amin(y)
    axes.plot(x_opt, y_opt, color='black', marker='o', markerfacecolor='white', markersize=12, clip_on=False)
    return im


def plot_2d(fig, error_analysis_parameters, score_vs_parameter_set, optimized_parameters, fitting_parameters, score_threshold):
    # Set the fitting parameters
    parameter1_id = error_analysis_parameters[0] 
    parameter2_id = error_analysis_parameters[1]
    # Set the values of the fitting parameters and the corresponding score values
    x1 = score_vs_parameter_set['parameters'][0] / const['fitting_parameters_scales'][parameter1_id.name]
    x2 = score_vs_parameter_set['parameters'][1] / const['fitting_parameters_scales'][parameter2_id.name]
    y = score_vs_parameter_set['score']
    # Interpolate the data points (x1, x2, y) on a regular grid (X, Y, Z)
    x1_min = np.min(x1)
    x1_max = np.max(x1)
    x2_min = np.min(x2)
    x2_max = np.max(x2)
    x1r = np.linspace(x1_min, x1_max, num=200) 
    x2r = np.linspace(x2_min, x2_max, num=200)
    X, Y = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]
    Z = griddata((x1, x2), y, (X, Y), method='linear')
    # Set the color code for Z
    cmin = np.amin(y) + score_threshold
    cmax = 2 * cmin
    # Plot the figure
    axes = fig.gca()
    im = axes.pcolor(X, Y, Z, cmap='jet_r', vmin=cmin, vmax=cmax)
    axes.set_xlim(np.amin(x1), np.amax(x1))
    axes.set_ylim(np.amin(x2), np.amax(x2))
    xlabel_text = const['fitting_parameters_labels'][parameter1_id.name][0] + \
                  r'$_{%d, %d}$' % (parameter1_id.spin_pair+1, parameter1_id.component+1) + ' ' + \
                  const['fitting_parameters_labels'][parameter1_id.name][1]
    ylabel_text = const['fitting_parameters_labels'][parameter2_id.name][0] + \
                  r'$_{%d, %d}$' % (parameter2_id.spin_pair+1, parameter2_id.component+1) + ' ' + \
                  const['fitting_parameters_labels'][parameter2_id.name][1]
    axes.set_xlabel(xlabel_text)
    axes.set_ylabel(ylabel_text)
    # Depict the optimized fitting parameter
    parameter1_index = parameter1_id.get_index(fitting_parameters['indices'])
    parameter2_index = parameter2_id.get_index(fitting_parameters['indices']) 
    x1_opt = optimized_parameters[parameter1_index] / const['fitting_parameters_scales'][parameter1_id.name]
    x2_opt = optimized_parameters[parameter2_index] / const['fitting_parameters_scales'][parameter2_id.name]
    axes.plot(x1_opt, x2_opt, color='black', marker='o', markerfacecolor='white', markersize=12, clip_on=False)
    return im


def plot_score_vs_parameters(error_analysis_parameters, score_vs_parameter_sets, optimized_parameters, fitting_parameters, score_threshold):
    ''' Plots the score as a function of one or two fitting parameters '''
    fig = plt.figure(figsize=[10,8], facecolor='w', edgecolor='w')  
    figsize = fig.get_size_inches()*fig.dpi
    num_subplots = len(error_analysis_parameters)
    layout = best_layout(figsize[0], figsize[1], num_subplots)
    for i in range(num_subplots):
        plt.subplot(layout[1], layout[0], i+1)
        dim = len(error_analysis_parameters[i])
        if (dim == 1):
            im = plot_1d(fig, error_analysis_parameters[i], score_vs_parameter_sets[i], optimized_parameters, fitting_parameters, score_threshold)
        elif (dim == 2):
            im = plot_2d(fig, error_analysis_parameters[i], score_vs_parameter_sets[i], optimized_parameters, fitting_parameters, score_threshold)
        else:
            print('The score cannot be yet plotted as function of three or more parameters!')
    plt.tight_layout()
    plt_set_fullscreen() 
    plt.subplots_adjust(bottom=0.10, top=0.90, right=0.80)
    cax = plt.axes([0.85, 0.3, 0.02, 0.4]) # left, bottom, width, height  
    plt.colorbar(im, cax=cax, orientation='vertical')
    plt.text(0.90, 1.05, r'$\mathit{\chi^2}$', transform=cax.transAxes)
    plt.draw()
    plt.pause(0.000001)
    return fig