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
from mathematics.rounding import ceil_with_precision, floor_with_precision
from mathematics.find_nearest import find_nearest
from supplement.definitions import const


markers = ["o", "s", "^", "p", "h", "*", "d", "v", "<", ">"]


def plot_1d_error_surface(axes, error_surface, optimized_model_parameters, error_analysis_parameters, 
                          fitting_parameters, chi2_minimum, chi2_threshold, distributions_are_multimodal):
    # Set the values of the fitting parameter and the corresponding chi2 values
    parameter_id = error_analysis_parameters[0]
    x = error_surface['parameters'][0] / const['model_parameter_scales'][parameter_id.name]
    y = error_surface['chi2']
    # Plot the 1d error surface
    im = axes.scatter(x, y, c=y, cmap='jet_r', vmin=chi2_minimum + chi2_threshold, vmax=chi2_minimum*1.5+chi2_threshold)
    if parameter_id.name in const['angle_parameter_names']:
        x_min, x_max = floor_with_precision(np.amin(x),0), ceil_with_precision(np.amax(x),0)
        axes.set_xlim(x_min, x_max)
        axes.set_xticks(np.linspace(x_min, x_max, 3))
    else:
        x_min, x_max = np.amin(x), np.amax(x)
        axes.set_xlim(x_min, x_max)
        axes.set_xticks(np.linspace(x_min, x_max, 3))
    if distributions_are_multimodal:
        xlabel_text = const['model_parameter_labels'][parameter_id.name][0] + r'$_{%d}$' % (parameter_id.component+1) + ' ' + const['model_parameter_labels'][parameter_id.name][1]
    else:
        xlabel_text = const['model_parameter_labels'][parameter_id.name][0] + ' ' + const['model_parameter_labels'][parameter_id.name][1]
    ylabel_text = r'$\mathit{\chi^2}$'
    axes.set_xlabel(xlabel_text)
    axes.set_ylabel(ylabel_text)
    # Depict the optimized fitting parameter
    for i in range(len(optimized_model_parameters)):
        parameter_index = parameter_id.get_index(fitting_parameters['indices'])
        x_opt = optimized_model_parameters[i][parameter_index] / const['model_parameter_scales'][parameter_id.name]
        idx_x_opt = find_nearest(x, x_opt)
        y_opt = y[idx_x_opt]
        if len(optimized_model_parameters) <= 10:
            axes.plot(x_opt, y_opt, color='black', marker=markers[i], markerfacecolor='white', clip_on=False)
        else:
            axes.plot(x_opt, y_opt, color='black', marker='o', markerfacecolor='white', clip_on=False)
    axes.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True) 
    # Make axes square
    xl, xh = axes.get_xlim()
    yl, yh = axes.get_ylim()
    axes.set_aspect((xh-xl)/(yh-yl))
    return im


def plot_2d_error_surface(axes, error_surface, optimized_model_parameters, error_analysis_parameters, 
                          fitting_parameters, chi2_minimum, chi2_threshold, distributions_are_multimodal):
    # Set the values of the fitting parameters and the corresponding chi2 values
    parameter1_id = error_analysis_parameters[0] 
    parameter2_id = error_analysis_parameters[1]
    x1 = error_surface['parameters'][0] / const['model_parameter_scales'][parameter1_id.name]
    x2 = error_surface['parameters'][1] / const['model_parameter_scales'][parameter2_id.name]
    y = error_surface['chi2']
    # Interpolate the data points (x1, x2, y) on a regular grid (X, Y, Z)
    size = int(np.sqrt(x1.size))*1j
    X, Y = np.mgrid[np.amin(x1):np.amax(x1):size, np.amin(x2):np.amax(x2):size]
    Z = griddata((x1, x2), y, (X, Y), method='nearest')
    # Plot the 2d error surface
    im = axes.pcolor(X, Y, Z, cmap='jet_r', vmin=chi2_minimum + chi2_threshold, vmax=chi2_minimum*1.5+chi2_threshold)
    if parameter1_id.name in const['angle_parameter_names']:
        x_min, x_max = floor_with_precision(np.amin(x1),0), ceil_with_precision(np.amax(x1),0)
        axes.set_xlim(x_min, x_max)
        axes.set_xticks(np.linspace(x_min, x_max, 3))
    else:
        x_min, x_max = np.amin(x1), np.amax(x1)
        axes.set_xlim(x_min, x_max)
        axes.set_xticks(np.linspace(x_min, x_max, 3))
    if parameter2_id.name in const['angle_parameter_names']:
        x_min, x_max = floor_with_precision(np.amin(x2),0), ceil_with_precision(np.amax(x2),0)
        axes.set_ylim(x_min, x_max)
        axes.set_yticks(np.linspace(x_min, x_max, 3))
    else:
        x_min, x_max = np.amin(x2), np.amax(x2)
        axes.set_ylim(x_min, x_max)
        axes.set_yticks(np.linspace(x_min, x_max, 3))
    if distributions_are_multimodal:
        xlabel_text = const['model_parameter_labels'][parameter1_id.name][0] + r'$_{%d}$' % (parameter1_id.component+1) + ' ' + const['model_parameter_labels'][parameter1_id.name][1]
        ylabel_text = const['model_parameter_labels'][parameter2_id.name][0] + r'$_{%d}$' % (parameter2_id.component+1) + ' ' + const['model_parameter_labels'][parameter2_id.name][1]
    else:
        xlabel_text = const['model_parameter_labels'][parameter1_id.name][0] + ' ' + const['model_parameter_labels'][parameter1_id.name][1]
        ylabel_text = const['model_parameter_labels'][parameter2_id.name][0] + ' ' + const['model_parameter_labels'][parameter2_id.name][1]
    axes.set_xlabel(xlabel_text)
    axes.set_ylabel(ylabel_text)
    # Depict the optimized fitting parameter
    for i in range(len(optimized_model_parameters)):
        parameter1_index = parameter1_id.get_index(fitting_parameters['indices'])
        parameter2_index = parameter2_id.get_index(fitting_parameters['indices']) 
        x1_opt = optimized_model_parameters[i][parameter1_index] / const['model_parameter_scales'][parameter1_id.name]
        x2_opt = optimized_model_parameters[i][parameter2_index] / const['model_parameter_scales'][parameter2_id.name]
        if len(optimized_model_parameters) <= 10:
            axes.plot(x1_opt, x2_opt, color='black', marker=markers[i], markerfacecolor='white', clip_on=False)
        else:
            axes.plot(x1_opt, x2_opt, color='black', marker='o', markerfacecolor='white', clip_on=False)
    # Make axes square
    xl, xh = axes.get_xlim()
    yl, yh = axes.get_ylim()
    axes.set_aspect((xh-xl)/(yh-yl))
    return im


def plot_error_surfaces(error_surfaces, error_surfaces_2d, optimized_model_parameters, error_analysis_parameters, fitting_parameters, chi2_minimum, chi2_thresholds):
    ''' Plots chi2 as a function of fitting parameter subsets '''  
    figsize = [10, 8]
    num_subplots = 0
    for i in range(len(error_analysis_parameters)):
        dim = len(error_analysis_parameters[i])
        if dim <= 2:
            num_subplots += 1
        else:
            num_subplots += dim * (dim - 1) // 2
    best_rcparams(num_subplots)
    layout = best_layout(figsize[0], figsize[1], num_subplots)
    fig = plt.figure(figsize=(figsize[0], figsize[1]), facecolor='w', edgecolor='w')
    if len(fitting_parameters['indices']['r_mean']) > 1:
        distributions_are_multimodal = True
    else:
        distributions_are_multimodal = False
    s = 1
    for i in range(len(error_analysis_parameters)):
        dim = len(error_analysis_parameters[i])
        chi2_threshold = chi2_thresholds[i]
        if dim == 1:
            if num_subplots == 1:
                axes = fig.gca()
            else:
                axes = fig.add_subplot(layout[0], layout[1], s)
                s += 1
            im = plot_1d_error_surface(axes, error_surfaces[i], optimized_model_parameters, error_analysis_parameters[i], 
                                       fitting_parameters, chi2_minimum, chi2_threshold, distributions_are_multimodal)
        elif dim == 2:
            if num_subplots == 1:
                axes = fig.gca()
            else:
                axes = fig.add_subplot(layout[0], layout[1], s)
                s += 1
            im = plot_2d_error_surface(axes, error_surfaces[i], optimized_model_parameters, error_analysis_parameters[i], 
                                       fitting_parameters, chi2_minimum, chi2_threshold, distributions_are_multimodal)
        else:
            c = 0
            for k in range(dim - 1):
                for l in range(k+1, dim):
                    parameters = [error_analysis_parameters[i][k], error_analysis_parameters[i][l]]
                    error_surface = error_surfaces_2d[i][c]
                    c += 1
                    if num_subplots == 1:
                        axes = fig.gca()
                    else:
                        axes = fig.add_subplot(layout[0], layout[1], s)
                        s += 1
                    im = plot_2d_error_surface(axes, error_surface, optimized_model_parameters, parameters, 
                                               fitting_parameters, chi2_minimum, chi2_threshold, distributions_are_multimodal)
    left = 0
    right = float(layout[1])/float(layout[1]+1)
    bottom = 0.5 * (1-right)
    top = 1 - bottom
    fig.tight_layout(rect=[left, bottom, right, top]) 
    cax = plt.axes([right+0.05, 0.5-0.5/float(layout[0])*0.5, 0.02, 1/float(layout[0])*0.5])
    cbar = plt.colorbar(im, cax=cax, orientation='vertical')
    cbar.formatter.set_powerlimits((0, 0))
    cbar.formatter.set_useMathText(True)
    cbar.ax.yaxis.set_offset_position('left') 
    plt.text(right-1.8, 1.05, r'$\mathit{\chi^2}$', transform=cax.transAxes)
    return fig