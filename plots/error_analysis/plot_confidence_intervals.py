import sys
import numpy as np
import plots.set_matplotlib
import matplotlib.pyplot as plt
from plots.best_layout import best_layout
from plots.plt_set_fullscreen import plt_set_fullscreen 
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
from supplement.definitions import const    


def plot_confidence_interval(fig, parameter_values, score_values, optimized_parameter_value, parameter_id, score_threshold, numerical_error):
    # Create a new parameter axis
    parameter_min = np.amin(parameter_values)
    parameter_max = np.amax(parameter_values)
    num_points = 100
    parameter_increment = (parameter_max - parameter_min) / float(num_points)
    parameter_axis = np.arange(parameter_min, parameter_max, parameter_increment)
    score_axis = np.zeros(num_points)
    indices_nonempty_bins = []
    for i in range(num_points):
        indices_parameter_axis = np.where(np.abs(parameter_values-parameter_axis[i]) < 0.5*parameter_increment)[0]
        if indices_parameter_axis != []:
            score_axis[i] = np.amin(score_values[indices_parameter_axis])
            indices_nonempty_bins.append(i)
    parameter_axis = parameter_axis[indices_nonempty_bins]
    score_axis = score_axis[indices_nonempty_bins]
    # Find the parameters ranges in which the score is within the confidence interval
    best_score = np.amin(score_axis)
    indices_confidence_interval = np.where(score_axis - best_score <= score_threshold)[0]
    confidence_interval_lower_bound = parameter_axis[indices_confidence_interval[0]]
    confidence_interval_upper_bound = parameter_axis[indices_confidence_interval[-1]]
    # Plot the figure
    x = parameter_axis
    y = score_axis
    # Set the color code for y
    cmin = np.amin(score_values) + score_threshold
    cmax = 2 * cmin
    axes = fig.gca()
    axes.scatter(x, y, c=y, cmap='jet_r', vmin=cmin, vmax=cmax)
    xlabel_text = const['fitting_parameters_labels'][parameter_id.name][0] + \
                  r'$_{%d, %d}$' % (parameter_id.spin_pair+1, parameter_id.component+1) + ' ' + \
                  const['fitting_parameters_labels'][parameter_id.name][1]
    ylabel_text = r'$\mathit{\chi^2}$'
    axes.set_xlabel(xlabel_text)
    axes.set_ylabel(ylabel_text)
    # Depict the confidence interval
    axes.axvspan(confidence_interval_lower_bound, confidence_interval_upper_bound, facecolor="lightgray", alpha=0.3, label="confidence interval")
    # Depict the optimized fitting parameter
    x_opt = optimized_parameter_value
    index_x_opt = min(range(len(x)), key=lambda i: abs(x[i]-x_opt))
    y_opt = y[index_x_opt]
    axes.plot(x_opt, y_opt, color='black', marker='o', markerfacecolor='white', markersize=12, clip_on=False, label = "fitting result")
    # Depict the score thresholds
    y1 = (best_score + score_threshold - numerical_error) * np.ones(x.size)
    y2 = (best_score + score_threshold) * np.ones(x.size)
    axes.plot(x, y1, 'm--', label = r'$\mathit{\chi^{2}_{min}}$ + $\mathit{\Delta\chi^{2}_{ci}}$')
    axes.plot(x, y2, 'k--', label = r'$\mathit{\chi^{2}_{min}}$ + $\mathit{\Delta\chi^{2}_{ci}}$ + $\mathit{\Delta\chi^{2}_{ne}}$')
    axes.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True) 


def plot_confidence_intervals(error_analysis_parameters, score_vs_parameter_sets, optimized_parameters, fitting_parameters, score_threshold, numerical_error):
    ''' Plots the confidence intervals of optimized fitting parameters '''
    fig = plt.figure(figsize=[10,8], facecolor='w', edgecolor='w')  
    figsize = fig.get_size_inches()*fig.dpi
    num_subplots = sum(len(i) for i in error_analysis_parameters)
    layout = best_layout(figsize[0], figsize[1], num_subplots)
    no_subplot = 1
    for i in range(len(error_analysis_parameters)):
        for j in range(len(error_analysis_parameters[i])):
            plt.subplot(layout[1], layout[0], no_subplot)
            no_subplot += 1
            parameter_id = error_analysis_parameters[i][j]
            parameter_values = score_vs_parameter_sets[i]['parameters'][j]  / const['fitting_parameters_scales'][parameter_id.name]
            score_values = score_vs_parameter_sets[i]['parameters'][j]
            parameter_index = parameter_id.get_index(fitting_parameters['indices']) 
            optimized_parameter_value = optimized_parameters[parameter_index] / const['fitting_parameters_scales'][parameter_id.name]
            plot_confidence_interval(fig, parameter_values, score_values, optimized_parameter_value, parameter_id, score_threshold, numerical_error)
    axes_list = fig.axes
    handles, labels = axes_list[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', frameon=False)
    fig.tight_layout()
    plt_set_fullscreen()
    fig.subplots_adjust(right=0.80)
    plt.draw()
    plt.pause(0.000001)
    return fig