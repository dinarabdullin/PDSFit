import sys
import numpy as np
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


def plot_error_profile(axes, error_profile, parameter_id, optimized_parameter_values, uncertainty_interval_bounds, 
                       fitting_parameters, chi2_minimum, chi2_threshold, distributions_are_multimodal):
    # Set the values of the fitting parameter and the corresponding chi2 values
    x = error_profile['parameter']  / const['model_parameter_scales'][parameter_id.name]
    y = error_profile['chi2']
    # Plot the chi2 vs fitting parameter
    axes.scatter(x, y, c=y, cmap='jet_r', vmin=chi2_minimum + chi2_threshold, vmax=chi2_minimum*1.5  + chi2_threshold, s=4)
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
    # Depict the confidence interval
    if uncertainty_interval_bounds != []:
        lower_bound = uncertainty_interval_bounds[0] / const['model_parameter_scales'][parameter_id.name]
        upper_bound = uncertainty_interval_bounds[1] / const['model_parameter_scales'][parameter_id.name]
        axes.axvspan(lower_bound, upper_bound, facecolor="lightgray", alpha=0.3, label="confidence\n interval")
    # Depict the optimized fitting parameter
    parameter_index = parameter_id.get_index(fitting_parameters['indices']) 
    for i in range(len(optimized_parameter_values)):
        x_opt = optimized_parameter_values[i][parameter_index] / const['model_parameter_scales'][parameter_id.name]
        idx_x_opt = find_nearest(x, x_opt)
        y_opt = y[idx_x_opt]
        if len(optimized_parameter_values) <= 10:
            axes.plot(x_opt, y_opt, color='black', marker=markers[i], markerfacecolor='white', clip_on=False)
        else:
            axes.plot(x_opt, y_opt, color='black', marker='o', markerfacecolor='white', clip_on=False)
    # Depict the chi2 thresholds
    threshold = (chi2_minimum + chi2_threshold) * np.ones(x.size)
    axes.plot(x, threshold, 'k--', label = r'$\mathit{\chi^{2}_{min}}$ + $\mathit{\Delta\chi^{2}}$')
    axes.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True) 
    # Make axes square
    xl, xh = axes.get_xlim()
    yl, yh = axes.get_ylim()
    axes.set_aspect((xh-xl)/(yh-yl))


def plot_error_profiles(error_profiles, optimized_model_parameters, error_analysis_parameters, fitting_parameters, 
                        model_parameter_uncertainty_interval_bounds, chi2_minimum, chi2_thresholds):
    ''' Plots chi2 as a function of individual fitting parameters '''
    figsize = [10, 8]
    num_subplots = sum(len(i) for i in error_analysis_parameters)
    best_rcparams(num_subplots)
    layout = best_layout(figsize[0], figsize[1], num_subplots)
    fig = plt.figure(figsize=(figsize[0], figsize[1]), facecolor='w', edgecolor='w')
    if len(fitting_parameters['indices']['r_mean']) > 1:
        distributions_are_multimodal = True
    else:
        distributions_are_multimodal = False
    c = 0
    for i in range(len(error_analysis_parameters)):
        for j in range(len(error_analysis_parameters[i])):
            parameter_id = error_analysis_parameters[i][j]
            error_profile = error_profiles[c]
            uncertainty_interval_bounds = model_parameter_uncertainty_interval_bounds[i][j]
            if num_subplots == 1:
                axes = fig.gca()
            else:
                axes = fig.add_subplot(layout[0], layout[1], c+1)
            chi2_threshold = chi2_thresholds[i]
            plot_error_profile(axes, error_profile,  parameter_id, optimized_model_parameters, model_parameter_uncertainty_interval_bounds[i][j], 
                               fitting_parameters, chi2_minimum, chi2_threshold, distributions_are_multimodal)
            c += 1
    left = 0
    right = float(layout[1])/float(layout[1]+1)
    bottom = 0.5 * (1-right)
    top = 1 - bottom
    fig.tight_layout(rect=[left, bottom, right, top])
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(right+0.01, 0.5), frameon=False) 
    return fig