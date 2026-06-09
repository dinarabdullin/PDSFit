import os
import sys
from datetime import datetime
import wx
import numpy as np
import scipy
import re
import copy
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['xtick.direction'] = 'out'
rcParams['ytick.direction'] = 'out'
rcParams['axes.facecolor']= 'white'
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = 'Arial'


def load_chi2_threshold(filepath):
    chi2_minimum, confidence_interval, threshold_theory, threshold_num_error = 0, 0, 0, 0
    with open(filepath, 'r') as file:
        text = file.read()
        # Regular expressions for the desired values
        chi2_minimum_match = re.search(r'Minimum chi-squared:\s*([\d.]+)', text)
        threshold_theory_match = re.search(r'Theoretical chi-squared threshold\s*\((\d+)-sigma\):\s*([\d.]+) \(1d\),\s*([\d.]+) \(2d\)', text)
        threshold_num_error_match = re.search(r'Numerical error contribution\s*\((\d+)-sigma\):\s*([\d.]+)', text)
        # Extracted values
        chi2_minimum = float(chi2_minimum_match.group(1)) if chi2_minimum_match else None
        confidence_interval = int(threshold_theory_match.group(1)) if threshold_theory_match else None
        threshold_theory_1d = float(threshold_theory_match.group(2)) if threshold_theory_match else None
        threshold_theory_2d = float(threshold_theory_match.group(3)) if threshold_theory_match else None
        threshold_theory = [threshold_theory_1d, threshold_theory_2d]
        threshold_num_error = float(threshold_num_error_match.group(2)) if threshold_num_error_match else None
    return chi2_minimum, confidence_interval, threshold_theory, threshold_num_error


def load_error_surfaces(input_path, fitting_parameters):
    error_surfaces = []
    for i in range(100):
        filepath = f'{input_path}/error_surface_{i}.dat'
        if os.path.exists(filepath):
            error_surface = load_error_surface(filepath, fitting_parameters)
            error_surfaces.append(error_surface)
    return error_surfaces


def load_error_surface(filepath, fitting_parameters):
    error_surface = {}
    parameter_subspace, x, y = [], [], []
    with open(filepath, 'r') as file:
        first_line = file.readline().strip()
        column_names = [first_line[i:i+30].strip() for i in range(0, len(first_line), 30)]
        for colum_name in column_names[:-1]:
            text = [x.strip() for x in colum_name.split(',')]
            for key in const['model_parameter_names_and_units']:  
                if text[0] == const['model_parameter_names_and_units'][key]:
                    name = key
            subtext = [x.strip() for x in text[1].split()]
            component = int(subtext[1]) - 1
            parameter = ParameterID(name, component)
            for fitting_parameter in fitting_parameters[name]:
                if parameter.name == fitting_parameter.name and parameter.component == fitting_parameter.component:
                    parameter.set_optimized(fitting_parameter.is_optimized())
                    parameter.set_index(fitting_parameter.get_index())      
            parameter_subspace.append(parameter)
        for next_line in file:
            row = [float(val) for val in next_line.strip().split()]
            x.append(row[:-1])
            y.append(row[-1])
        x = np.transpose(np.array(x))
        y = np.array(y)
        for i, parameter in enumerate(parameter_subspace):
            x[i,:] *= const['model_parameter_scales'][parameter.name]
            parameter_subspace[i].set_range([np.amin(x[i,:]), np.amax(x[i,:])])
        error_surface['par'] = parameter_subspace
        error_surface['x'] = x
        error_surface['y'] = y
    return error_surface
    
    
def compute_theoretical_chi2_thresholds(degrees_of_freedom, confidence_interval):
        chi2_thresholds_theory = []
        for v in degrees_of_freedom:
            chi2_threshold = 0.0
            if v == 1:
                chi2_threshold = float(confidence_interval)**2
            else:
                p = 1.0 - scipy.stats.chi2.sf(float(confidence_interval)**2, 1)
                chi2_threshold = scipy.stats.chi2.ppf(p, int(v))
            chi2_thresholds_theory.append(chi2_threshold)
        return np.array(chi2_thresholds_theory)


def get_user_input(question, answer_datatype = str, default_answer = '', answer_options = {}):
    '''
    Uses Q&A to get the user input.
    
    Arguments:
    question -- A question addressed to the user.
    answer_datatype -- The expected data type of an answer (atr, int, float).
    default_answer - A default answer.
    answer_options -- A dictionary with possible anwers (optinal).
    
    Returns:
    accepted_answer -- The answer provided by the user.
    '''
    var = input(question)
    if not var:
        if answer_options:
            accepted_answer = answer_options[default_answer]
        else:
            accepted_answer = default_answer
    else:
        if isinstance(answer_datatype, str):
            entered_answer = var
            if answer_options:
                if entered_answer in answer_options:
                    accepted_answer = answer_options[entered_answer]
                else:
                    raise ValueError('Unexpected answer!')
                    sys.exit(1)
            else:
                accepted_answer = entered_answer
        else:
            try:
                entered_answer = answer_datatype(var)
                if answer_options:
                    if entered_answer in answer_options:
                        accepted_answer = answer_options[entered_answer]
                    else:
                        raise ValueError('Unexpected answer!')
                        sys.exit(1)
                else:
                    accepted_answer = entered_answer
            except (ValueError, TypeError):
                raise ValueError('Unexpected answer!')
                sys.exit(1)
    sys.stdout.write('User input: {0:s}\n'.format(str(accepted_answer)))
    return accepted_answer

def update_fontsize(fig, new_fontsize, layout):
    # Update all axes-level fonts
    for ax in fig.get_axes():
        # Title and labels
        ax.title.set_fontsize(new_fontsize)
        ax.xaxis.label.set_fontsize(new_fontsize)
        ax.yaxis.label.set_fontsize(new_fontsize)
        # Tick labels
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(new_fontsize)
        # Legend
        legend = ax.get_legend()
        if legend:
            for text in legend.get_texts():
                text.set_fontsize(new_fontsize)
        # Text annotations within the axes
        for text in ax.texts:
            text.set_fontsize(new_fontsize)
        # Colorbar
        try:
            ax.tick_params(labelsize=new_fontsize)
            offset_text = ax.yaxis.get_offset_text()
            offset_text.set_fontsize(new_fontsize)
            ax.yaxis.label.set_fontsize(new_fontsize)
        except Exception as e:
            pass
    # Figure-level texts
    for text in fig.texts:
        text.set_fontsize(new_fontsize)
    left = 0
    right = float(layout[1]) / float(layout[1] + 1)
    bottom = 0.5 * (1 - right)
    top = 1 - bottom
    fig.tight_layout(rect = [left, bottom, right, top]) 
    return fig


def plot_error_surfaces(
    error_surfaces, chi2_minimum, chi2_thresholds, optimized_model_parameters, 
    fitting_parameters, show_uncertainty_interval = False
    ):
    if len(fitting_parameters["r_mean"]) > 1:
        multimodal_distributions = True
    else:
        multimodal_distributions = False
    num_subplots = 0
    for error_surface in error_surfaces:
        if len(error_surface["par"]) <= 2:
            num_subplots += 1
    figsize = [10, 8]
    best_rcparams(num_subplots)
    rcParams['xtick.major.size'] = 4
    rcParams['ytick.major.size'] = 4
    rcParams['font.size'] = 17
    rcParams['lines.markersize'] = 10
    layout = best_layout(figsize[0], figsize[1], num_subplots)
    fig = plt.figure(
        figsize = (figsize[0], figsize[1]),
        facecolor = "w",
        edgecolor = "w"
        )
    n_subplot = 1
    for error_surface in error_surfaces:
        dim = len(error_surface["par"])
        if dim == 1:
            if num_subplots == 1:
                axes = fig.gca()
            else:
                axes = fig.add_subplot(layout[0], layout[1], n_subplot)
            im = plot_error_surface_1d(
                axes, copy.deepcopy(error_surface), chi2_minimum, chi2_thresholds[0], 
                optimized_model_parameters, multimodal_distributions, show_uncertainty_interval
                )
            n_subplot += 1
        elif dim == 2:
            if num_subplots == 1:
                axes = fig.gca()
            else:
                axes = fig.add_subplot(layout[0], layout[1], n_subplot)
            im = plot_error_surface_2d(
                axes, copy.deepcopy(error_surface), chi2_minimum, chi2_thresholds[1], 
                optimized_model_parameters, multimodal_distributions
                )
            n_subplot += 1
        else:
            pass
    # Rescale figure axes to add a colorbar
    left = 0
    right = float(layout[1]) / float(layout[1] + 1)
    bottom = 0.5 * (1 - right)
    top = 1 - bottom
    fig.tight_layout(rect = [left, bottom, right, top])
    # Add a colorbar
    cax = plt.axes([right + 0.05, 0.5 - 0.5 / float(layout[0]) * 0.5, 0.02, 1 / float(layout[0]) * 0.5])
    cbar = plt.colorbar(im, cax = cax, orientation = "vertical")
    cbar.formatter.set_powerlimits((0, 0))
    cbar.formatter.set_useMathText(True)
    cbar.ax.yaxis.set_offset_position("left") 
    plt.text(right - 1.8, 1.05, r"$\mathit{\chi^2}$", transform = cax.transAxes)
    return fig


if __name__ == '__main__':
    # Load PDSFit modules
    # Load the data from the folder
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(parent_dir)
    from fitting.parameter_id import ParameterID
    from error_analysis.load_optimized_models import load_optimized_model
    from error_analysis.error_analyzer import ErrorAnalyzer
    from output.fitting.save_model_parameters import print_model_parameters
    from supplement.definitions import const
    from output.data_saver import DataSaver
    from plots.plotter import Plotter
    from plots.error_analysis.plot_error_surfaces import plot_error_surface_1d, plot_error_surface_2d
    from plots.best_layout import best_layout
    from plots.set_matplotlib import best_rcparams
    
    # Open the folder with the PDS results
    app = wx.App(False)
    dialog = wx.DirDialog(None, 'Select a folder with PDS results', style=wx.DD_DEFAULT_STYLE)
    if dialog.ShowModal() == wx.ID_OK:
        input_path = dialog.GetPath()
        if os.path.exists(input_path) and os.path.isdir(input_path):
            date_str = datetime.now().strftime('%Y_%m_%d')
            output_folder_name = f'{date_str}_error_analysis'
            os.makedirs(os.path.join(input_path, output_folder_name), exist_ok=True)
    else:
        print('No folder selected.')
    dialog.Destroy()
    
    # Read the PDSFit results
    # Optimized fitting parameters
    fitting_parameters, optimized_model, errors = load_optimized_model(input_path + '/fitting_parameters.dat')
    # Error surfaces
    all_error_surfaces = load_error_surfaces(input_path, fitting_parameters)
    num_parameter_subspaces = len(all_error_surfaces)
    # Chi2 thresholds
    chi2_minimum, confidence_interval, chi2_threshold_theory, chi2_threshold_num_error = load_chi2_threshold(input_path + '/logfile.log')
    chi2_threshold_num_error_1sigma = chi2_threshold_num_error / float(confidence_interval)
    # Print
    print('\n=== Previous chi-squared threshold ===')
    print(f'Minimum chi-squared: {chi2_minimum:.1f}')
    print(f'Theoretical chi-squared threshold ({confidence_interval}-sigma): {chi2_threshold_theory[0]:.1f} (1d), {chi2_threshold_theory[1]} (2d)')
    print(f'Numerical error contribution ({confidence_interval}-sigma): {chi2_threshold_num_error:.1f}')
    print_model_parameters(optimized_model, fitting_parameters, errors)
    
    # Set the new confidence interval
    print('\n=== New chi-squared threshold ===')
    new_confidence_interval = get_user_input('Choose the new confidence interval [default value: 2]: ', int, 2, {})
    if new_confidence_interval != confidence_interval:
        num_parameters = len(optimized_model)
        max_dim_error_surface = max(len(error_surface['par']) for error_surface in all_error_surfaces)
        degrees_of_freedom = np.arange(num_parameters, num_parameters - max_dim_error_surface, -1)
        chi2_thresholds_theory = compute_theoretical_chi2_thresholds(degrees_of_freedom, new_confidence_interval)
        chi2_threshold_num_error = float(new_confidence_interval) * chi2_threshold_num_error_1sigma
        chi2_thresholds = chi2_thresholds_theory + chi2_threshold_num_error
        print(f'Minimum chi-squared: {chi2_minimum: .1f}')
        print(f'Theoretical chi-squared threshold ({new_confidence_interval}-sigma): {chi2_thresholds_theory[0]:.1f} (1d), {chi2_thresholds_theory[1]:.1f} (2d)')
        print(f'Numerical error contribution ({new_confidence_interval}-sigma): {chi2_threshold_num_error:.1f}')
    else:
        new_confidence_interval = confidence_interval
        num_parameters = len(optimized_model)
        max_dim_error_surface = max(len(error_surface['par']) for error_surface in all_error_surfaces)
        degrees_of_freedom = np.arange(num_parameters, num_parameters - max_dim_error_surface, -1)
        chi2_thresholds_theory = compute_theoretical_chi2_thresholds(degrees_of_freedom, new_confidence_interval)
        chi2_thresholds = chi2_thresholds_theory + chi2_threshold_num_error
       
    # Calculate error surfaces and the errors of individual parameters
    error_analyzer = ErrorAnalyzer()
    error_analyzer.set_intrinsic_parameters({
        'confidence_interval'        : new_confidence_interval,
        'samples_per_parameter'      : int(np.power(len(all_error_surfaces[0]['y']), 1.0 / len(all_error_surfaces[0]['par']))),
        'samples_numerical_error'    : 1,
        'filepath_fitting_parameters': ''
    })
    errors_model_parameters = error_analyzer.init_errors_model_parameters(optimized_model)
    all_error_surfaces_1d, all_error_surfaces_2d = [], []
    for error_surface in all_error_surfaces:
        num_parameters = len(error_surface['par'])
        if num_parameters > 2:
            error_surfaces_2d = error_analyzer.compute_2d_error_surfaces(error_surface)
            all_error_surfaces_2d.extend(error_surfaces_2d)
        if num_parameters > 1:
            error_surfaces_1d = error_analyzer.compute_1d_error_surfaces(error_surface)
        else:
            error_surfaces_1d = [error_surface]
        for error_surface_1d in error_surfaces_1d:
             error_surface_1d = error_analyzer.reset_minimum_chi2(error_surface_1d, chi2_minimum, optimized_model)
        all_error_surfaces_1d.extend(error_surfaces_1d)
        # Compute the errors of model parameters
        for error_surface_1d in error_surfaces_1d:
            error_model_parameter = error_analyzer.compute_model_parameter_error(optimized_model, error_surface_1d, chi2_thresholds, chi2_minimum)
            errors_model_parameters = error_analyzer.update_errors_model_parameters(error_surface_1d['par'][0], error_model_parameter, errors_model_parameters)
    print_model_parameters(optimized_model, fitting_parameters, errors_model_parameters)
    
    # Save the error analysis data
    data_saver = DataSaver(save_data=True, save_figures=True)
    data_saver.create_output_directory(parent_directory=input_path, filepath_config='x/error_analysis.xxx')
    data_saver.save_model_parameters([optimized_model], 0, fitting_parameters, errors_model_parameters)  

    # Plot the error analysis data
    print('\n=== Plotting error analysis results ===')
    #new_fontsize = get_user_input('Prefered font size [default value: None]: ', int, None, {})
    
    fig = plot_error_surfaces(
        all_error_surfaces + all_error_surfaces_2d, chi2_minimum, chi2_thresholds, [optimized_model], fitting_parameters, False
        )
    #if new_fontsize:
    #    layout = best_layout(10, 8, len(all_error_surfaces + all_error_surfaces_2d))   
    #    fig = update_fontsize(fig, new_fontsize, layout)
    filepath = data_saver.directory + "error_surfaces.png"
    fig.savefig(filepath, format = "png", dpi = 600)
    
    fig = plot_error_surfaces(
        all_error_surfaces_1d, chi2_minimum, chi2_thresholds, [optimized_model], fitting_parameters, False
        )
    #if new_fontsize:
    #    layout = best_layout(10, 8, len(all_error_surfaces_1d))   
    #    fig = update_fontsize(fig, new_fontsize, layout)
    filepath = data_saver.directory + "error_surfaces_1d.png"
    fig.savefig(filepath, format = "png", dpi = 600)