import sys
import numpy as np
from scipy.interpolate import griddata
import plots.set_matplotlib
from plots.set_matplotlib import best_rcparams
import matplotlib.pyplot as plt
from plots.best_layout import best_layout
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
from supplement.definitions import const    


def plot_1d_error_surface(axes, score_vs_parameter_subset, error_analysis_parameters, fitting_parameters, optimized_parameters, score_threshold):
    # Set the values of the fitting parameter and the corresponding score values
    parameter_id = error_analysis_parameters[0]
    x = score_vs_parameter_subset['parameters'][0] / const['fitting_parameters_scales'][parameter_id.name]
    y = score_vs_parameter_subset['score']
    # Set the color ranges
    cmin = np.amin(y) + score_threshold
    cmax = 2 * cmin
    # Plot the 1d error surface
    im = axes.scatter(x, y, c=y, cmap='jet_r', vmin=cmin, vmax=cmax)
    axes.set_xlim(round(np.amin(x),1), round(np.amax(x),1))
    xlabel_text = const['fitting_parameters_labels'][parameter_id.name][0] + r'$_{%d, %d}$' % (parameter_id.spin_pair+1, parameter_id.component+1) + ' ' + const['fitting_parameters_labels'][parameter_id.name][1]
    ylabel_text = r'$\mathit{\chi^2}$'
    axes.set_xlabel(xlabel_text)
    axes.set_ylabel(ylabel_text)
    # Depict the optimized fitting parameter
    parameter_index = parameter_id.get_index(fitting_parameters['indices']) 
    x_opt = optimized_parameters[parameter_index] / const['fitting_parameters_scales'][parameter_id.name]
    y_opt = np.amin(y)
    axes.plot(x_opt, y_opt, color='black', marker='o', markerfacecolor='white', clip_on=False)
    # Make axes square
    xl, xh = axes.get_xlim()
    yl, yh = axes.get_ylim()
    axes.set_aspect((xh-xl)/(yh-yl))
    return im


def plot_2d_error_surface(axes, score_vs_parameter_subset, error_analysis_parameters, fitting_parameters, optimized_parameters, score_threshold):
    # Set the values of the fitting parameters and the corresponding score values
    parameter1_id = error_analysis_parameters[0] 
    parameter2_id = error_analysis_parameters[1]
    x1 = score_vs_parameter_subset['parameters'][0] / const['fitting_parameters_scales'][parameter1_id.name]
    x2 = score_vs_parameter_subset['parameters'][1] / const['fitting_parameters_scales'][parameter2_id.name]
    y = score_vs_parameter_subset['score']
    # Interpolate the data points (x1, x2, y) on a regular grid (X, Y, Z)
    x1_min = np.min(x1)
    x1_max = np.max(x1)
    x2_min = np.min(x2)
    x2_max = np.max(x2)
    x1r = np.linspace(x1_min, x1_max, num=200) 
    x2r = np.linspace(x2_min, x2_max, num=200)
    X, Y = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]
    Z = griddata((x1, x2), y, (X, Y), method='linear')
    # Set the color ranges for Z
    cmin = np.amin(y) + score_threshold
    cmax = 2 * cmin
    # Plot the 2d error surface
    #extent = x1_min, x1_max, x2_min, x2_max
    #im = plt.imshow(Z, cmap='jet_r', interpolation='nearest', extent=extent, aspect=abs((extent[1]-extent[0])/(extent[3]-extent[2])), vmin=cmin, vmax=cmax)
    im = axes.pcolor(X, Y, Z, cmap='jet_r', vmin=cmin, vmax=cmax)
    axes.set_xlim(np.amin(x1), np.amax(x1))
    axes.set_ylim(np.amin(x2), np.amax(x2))
    xlabel_text = const['fitting_parameters_labels'][parameter1_id.name][0] + r'$_{%d, %d}$' % (parameter1_id.spin_pair+1, parameter1_id.component+1) + ' ' + const['fitting_parameters_labels'][parameter1_id.name][1]
    ylabel_text = const['fitting_parameters_labels'][parameter2_id.name][0] + r'$_{%d, %d}$' % (parameter2_id.spin_pair+1, parameter2_id.component+1) + ' ' + const['fitting_parameters_labels'][parameter2_id.name][1]
    axes.set_xlabel(xlabel_text)
    axes.set_ylabel(ylabel_text)
    # Depict the optimized fitting parameter
    parameter1_index = parameter1_id.get_index(fitting_parameters['indices'])
    parameter2_index = parameter2_id.get_index(fitting_parameters['indices']) 
    x1_opt = optimized_parameters[parameter1_index] / const['fitting_parameters_scales'][parameter1_id.name]
    x2_opt = optimized_parameters[parameter2_index] / const['fitting_parameters_scales'][parameter2_id.name]
    axes.plot(x1_opt, x2_opt, color='black', marker='o', markerfacecolor='white', clip_on=False)
    # Make axes square
    xl, xh = axes.get_xlim()
    yl, yh = axes.get_ylim()
    axes.set_aspect((xh-xl)/(yh-yl))
    return im


def plot_error_surfaces(score_vs_parameter_subsets, error_analysis_parameters, fitting_parameters, optimized_parameters, score_threshold):
    ''' Plots the score as a function of one or two fitting parameters '''  
    figsize = [10, 8]
    num_subplots = len(error_analysis_parameters)
    best_rcparams(num_subplots)
    layout = best_layout(figsize[0], figsize[1], num_subplots)
    fig = plt.figure(figsize=(figsize[0], figsize[1]), facecolor='w', edgecolor='w')
    for i in range(num_subplots):
        dim = len(error_analysis_parameters[i])
        if num_subplots == 1:
            axes = fig.gca()
        else:
            axes = fig.add_subplot(layout[0], layout[1], i+1)
        if (dim == 1):
            im = plot_1d_error_surface(axes, score_vs_parameter_subsets[i], error_analysis_parameters[i], fitting_parameters, optimized_parameters, score_threshold)
        elif (dim == 2):
            im = plot_2d_error_surface(axes, score_vs_parameter_subsets[i], error_analysis_parameters[i], fitting_parameters, optimized_parameters, score_threshold)
        else:
            print('The score cannot be yet plotted as function of three or more parameters!')      
    left = 0
    right = float(layout[1])/float(layout[1]+1)
    bottom = 0.5 * (1-right)
    top = 1 - bottom
    plt.tight_layout(rect=[left, bottom, right, top]) 
    cax = plt.axes([right+0.05, 0.3, 0.02, 0.4])
    plt.colorbar(im, cax=cax, orientation='vertical')
    plt.text(right+0.1, 1.05, r'$\mathit{\chi^2}$', transform=cax.transAxes)
    plt.draw()
    plt.pause(0.000001)
    return fig