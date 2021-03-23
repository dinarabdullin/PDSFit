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
    # Plot the score vs fitting parameter
    cmin = np.amin(score_values) + score_threshold
    cmax = 2 * cmin
    axes = fig.gca()
    axes.scatter(parameter_values, score_values, c=score_values, cmap='jet_r', vmin=cmin, vmax=cmax)
    xlabel_text = const['fitting_parameters_labels'][parameter_id.name][0] + \
                  r'$_{%d, %d}$' % (parameter_id.spin_pair+1, parameter_id.component+1) + ' ' + \
                  const['fitting_parameters_labels'][parameter_id.name][1]
    ylabel_text = r'$\mathit{\chi^2}$'
    axes.set_xlabel(xlabel_text)
    axes.set_ylabel(ylabel_text)
    # Depict the confidence interval
    best_score = np.amin(score_values)
    indices_confidence_interval = np.where(score_values - best_score <= score_threshold)[0]
    confidence_interval_lower_bound = parameter_values[indices_confidence_interval[0]]
    confidence_interval_upper_bound = parameter_values[indices_confidence_interval[-1]]
    axes.axvspan(confidence_interval_lower_bound, confidence_interval_upper_bound, facecolor="lightgray", alpha=0.3, label="confidence interval")
    # Depict the optimized fitting parameter
    index_optimized_parameter = min(range(len(parameter_values)), key=lambda i: abs(parameter_values[i]-optimized_parameter_value))
    minimal_score = score_values[index_optimized_parameter]
    axes.plot(optimized_parameter_value, minimal_score, color='black', marker='o', markerfacecolor='white', markersize=12, clip_on=False, label = "fitting result")
    # Depict the score thresholds
    threshold1 = (best_score + score_threshold - numerical_error) * np.ones(parameter_values.size)
    threshold2 = (best_score + score_threshold) * np.ones(parameter_values.size)
    axes.plot(parameter_values, threshold1, 'm--', label = r'$\mathit{\chi^{2}_{min}}$ + $\mathit{\Delta\chi^{2}_{ci}}$')
    axes.plot(parameter_values, threshold2, 'k--', label = r'$\mathit{\chi^{2}_{min}}$ + $\mathit{\Delta\chi^{2}_{ci}}$ + $\mathit{\Delta\chi^{2}_{ne}}$')
    axes.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True) 


def plot_confidence_intervals(score_vs_parameters, error_analysis_parameters, fitting_parameters, optimized_parameters, score_threshold, numerical_error):
    ''' Plots the confidence intervals of optimized fitting parameters '''
    fig = plt.figure(figsize=(10,8), facecolor='w', edgecolor='w')  
    figsize = fig.get_size_inches()*fig.dpi
    num_subplots = sum(len(i) for i in error_analysis_parameters)
    layout = best_layout(figsize[0], figsize[1], num_subplots)
    counter = 0
    for i in range(len(error_analysis_parameters)):
        for j in range(len(error_analysis_parameters[i])):
            plt.subplot(layout[1], layout[0], counter+1)
            parameter_id = error_analysis_parameters[i][j]
            score_vs_parameter = score_vs_parameters[counter]
            parameter_values = score_vs_parameter['parameter']  / const['fitting_parameters_scales'][parameter_id.name]
            score_values = score_vs_parameter['score']
            parameter_index = parameter_id.get_index(fitting_parameters['indices']) 
            optimized_parameter_value = optimized_parameters[parameter_index] / const['fitting_parameters_scales'][parameter_id.name]
            plot_confidence_interval(fig, parameter_values, score_values, optimized_parameter_value, parameter_id, score_threshold, numerical_error)
            counter += 1
    axes_list = fig.axes
    handles, labels = axes_list[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', frameon=False)
    fig.tight_layout()
    plt_set_fullscreen()
    fig.subplots_adjust(right=0.80)
    plt.draw()
    plt.pause(0.000001)
    return fig