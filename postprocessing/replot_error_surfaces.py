import os
import io
import sys
import wx
import numpy as np
from textwrap import wrap
from scipy.interpolate import griddata
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import matplotlib
if os.name == 'posix' and "DISPLAY" not in os.environ:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt


def best_rcparams(n):
    from matplotlib import rcParams
    rcParams['xtick.direction'] = 'out'
    rcParams['ytick.direction'] = 'out'
    rcParams['axes.facecolor']= 'white'
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = 'Arial'
    ''' Adjusts the matplotlib's rcParams in dependence of subplots number '''
    if   n == 1:
        rcParams['lines.linewidth'] = 2
        rcParams['xtick.major.size'] = 8
        rcParams['xtick.major.width'] = 1.5
        rcParams['ytick.major.size'] = 8
        rcParams['ytick.major.width'] = 1.5
        rcParams['font.size'] = 18
        rcParams['lines.markersize'] = 10
    elif n >= 2 and n < 4:
        rcParams['lines.linewidth'] = 1.5
        rcParams['xtick.major.size'] = 4
        rcParams['xtick.major.width'] = 1.5
        rcParams['ytick.major.size'] = 4
        rcParams['ytick.major.width'] = 1
        rcParams['font.size'] = 14
        rcParams['lines.markersize'] = 10
    elif n >= 4 and n < 8:
        rcParams['lines.linewidth'] = 1
        rcParams['xtick.major.size'] = 4
        rcParams['xtick.major.width'] = 1
        rcParams['ytick.major.size'] = 4
        rcParams['ytick.major.width'] = 1
        rcParams['font.size'] = 12
        rcParams['lines.markersize'] = 8
    elif n >= 9 and n < 13:
        rcParams['lines.linewidth'] = 1
        rcParams['xtick.major.size'] = 4
        rcParams['xtick.major.width'] = 1
        rcParams['ytick.major.size'] = 4
        rcParams['ytick.major.width'] = 1
        rcParams['font.size'] = 10
        rcParams['lines.markersize'] = 6
    elif n >= 13:
        rcParams['lines.linewidth'] = 0.5
        rcParams['xtick.major.size'] = 4
        rcParams['xtick.major.width'] = 0.5
        rcParams['ytick.major.size'] = 4
        rcParams['ytick.major.width'] = 0.5
        rcParams['font.size'] = 8
        rcParams['lines.markersize'] = 4


def best_square_size(x, y, n):
    '''
    Given a rectangle with width and height, fill it with n squares of equal size such 
    that the squares cover as much of the rectangle's area as possible. 
    The size of a single square should be returned.
    Source: https://math.stackexchange.com/questions/466198/algorithm-to-get-the-maximum-size-of-n-squares-that-fit-into-a-rectangle-with-a
    '''
    x, y, n = float(x), float(y), float(n)
    px = np.ceil(np.sqrt(n * x / y))
    if np.floor(px * y / x) * px  < n:
            sx = y / np.ceil(px * y / x)
    else:
            sx = x/px
    py = np.ceil(np.sqrt(n * y / x))
    if np.floor(py * x / y) * py < n:
            sy = x / np.ceil(x * py / y)
    else:
            sy = y / py
    return max(sx,sy)


def best_layout(w, h, n):
    ''' Find the best layout of multiple subplots for a given screen size'''
    a = best_square_size(w, h, n)
    n_row = int(h/a)
    n_col = int(w/a)
    if n_row * n_col > n:
        if (n_row-1) * n_col >= n:
            return [n_row-1, n_col]
        elif n_row * (n_col-1) >= n:
            return [n_row, n_col-1]
        else:
            return [n_row, n_col]
    else:
        return [n_row, n_col]


const = {}
const['fitting_parameters_names'] = [
    'r_mean',
    'r_width',
    'xi_mean',
    'xi_width',
    'phi_mean',
    'phi_width',
    'alpha_mean',
    'alpha_width',
    'beta_mean',
    'beta_width',
    'gamma_mean',
    'gamma_width',
    'rel_prob',
    'j_mean',
    'j_width'
    ]  
const['fitting_parameters_labels'] = {
	'r_mean'      : [r'$\langle\mathit{r}\rangle$', '(nm)'],
	'r_width'     : [r'$\mathit{\Delta r}$', '(nm)'], 
	'xi_mean'     : [r'$\langle\mathit{\xi}\rangle$', '$^\circ$'],
	'xi_width'    : [r'$\mathit{\Delta\xi}$', '$^\circ$'], 
	'phi_mean'    : [r'$\langle\mathit{\varphi}\rangle$', '$^\circ$'], 
	'phi_width'   : [r'$\mathit{\Delta\varphi}$', '$^\circ$'],
    'alpha_mean'  : [r'$\langle\mathit{\alpha}\rangle$', '$^\circ$'],
    'alpha_width' : [r'$\mathit{\Delta\alpha}$', '$^\circ$'], 
    'beta_mean'   : [r'$\langle\mathit{\beta}\rangle$', '$^\circ$'],
    'beta_width'  : [r'$\mathit{\Delta\beta}$', '$^\circ$'], 
    'gamma_mean'  : [r'$\langle\mathit{\gamma}\rangle$', '$^\circ$'],
    'gamma_width' : [r'$\mathit{\Delta\gamma}$', '$^\circ$'],
    'rel_prob'    : [r'Relative weight'],
    'j_mean'      : [r'$\langle\mathit{J}\rangle$', '(MHz)'],
    'j_width'     : [r'$\mathit{\Delta J}$', '(MHz)']
    }    


class ParameterID:
    ''' ID of a fitting parameter ''' 

    def __init__(self, name, spin_pair, component):
        self.name = name
        self.spin_pair = spin_pair
        self.component = component
    
    def get_index(self, fitting_parameters_indices):
        return fitting_parameters_indices[self.name][self.spin_pair][self.component].index
    
    def is_optimized(self, fitting_parameters_indices):
        return fitting_parameters_indices[self.name][self.spin_pair][self.component].optimize


def get_path(message):
    app = wx.App(None) 
    dialog = wx.FileDialog(None, message, wildcard='*.*', style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
    if dialog.ShowModal() == wx.ID_OK:
        path = dialog.GetPath()
    return path


def load_error_analysis_data(directory):
    # Find all error analysis files
    error_analysis_files = []
    c = 1
    while True:
        filename = directory + 'score_vs_parameters_' + str(c) + '.dat'
        c += 1
        if os.path.exists(filename):
            error_analysis_files.append(filename)
        else:
            break
    n_files = len(error_analysis_files)
    # Set the error analysis parameters
    error_analysis_parameters = []
    for i in range(n_files):
        file = open(error_analysis_files[i], 'r')
        head = str(file.readline())
        column_names = wrap(head, 20)
        n_parameters = len(column_names) - 1
        subset_error_analysis_parameters = []
        for j in range(n_parameters):
            parameter_name = column_names[j].split()
            name = parameter_name[0].strip()
            spin_pair = int(parameter_name[1]) - 1 
            component = int(parameter_name[2]) - 1
            name_found = False 
            for item in const['fitting_parameters_names']:    
                if name == item:
                    name_found = True
            if not name_found:
                raise ValueError('Unknown parameter name was found!')
                sys.exit(1)
            parameter_id = ParameterID(name, spin_pair, component)
            subset_error_analysis_parameters.append(parameter_id)
            file.close()
        error_analysis_parameters.append(subset_error_analysis_parameters)    
    # Read the error analysis data
    score_vs_parameter_subsets = []
    for i in range(n_files):
        file = open(error_analysis_files[i], 'r')
        n_points = len(file.readlines()) - 1
        file.close()
        n_parameters = len(error_analysis_parameters[i])
        score_vs_parameter_subset = {}
        score_vs_parameter_subset['parameters'] = np.zeros([n_parameters,n_points])
        score_vs_parameter_subset['score'] = np.zeros(n_points)
        file = open(error_analysis_files[i], 'r')
        next(file)
        c = 0
        for line in file:
            data = wrap(line, 20)
            for j in range(n_parameters):
                score_vs_parameter_subset['parameters'][j][c] = float(data[j])
            score_vs_parameter_subset['score'][c] = float(data[-1])
            c += 1
        score_vs_parameter_subsets.append(score_vs_parameter_subset)
        file.close()
    return [error_analysis_parameters, score_vs_parameter_subsets]


def plot_1d_error_surface(axes, score_vs_parameter_subset, error_analysis_parameters, optimized_parameters, score_threshold):
    # Set the values of the fitting parameter and the corresponding score values
    parameter_id = error_analysis_parameters[0]
    x = score_vs_parameter_subset['parameters'][0]
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
    if optimized_parameters:
        x_opt = optimized_parameters[0]
        y_opt = np.amin(y)
        axes.plot(x_opt, y_opt, color='black', marker="^", markerfacecolor='white', clip_on=False)
    # Make axes square
    xl, xh = axes.get_xlim()
    yl, yh = axes.get_ylim()
    axes.set_aspect((xh-xl)/(yh-yl))
    return im


def plot_2d_error_surface(axes, score_vs_parameter_subset, error_analysis_parameters, optimized_parameters, score_threshold):
    # Set the values of the fitting parameters and the corresponding score values
    parameter1_id = error_analysis_parameters[0] 
    parameter2_id = error_analysis_parameters[1]
    x1 = score_vs_parameter_subset['parameters'][0]
    x2 = score_vs_parameter_subset['parameters'][1]
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
    # Depict the optimized fitting parameters
    if optimized_parameters:
        x1_opt = optimized_parameters[0]
        x2_opt = optimized_parameters[1]
        axes.plot(x1_opt, x2_opt, color='black', marker="^", markerfacecolor='white', clip_on=False)
    # Make axes square
    xl, xh = axes.get_xlim()
    yl, yh = axes.get_ylim()
    axes.set_aspect((xh-xl)/(yh-yl))
    return im


def plot_error_surfaces(score_vs_parameter_subsets, error_analysis_parameters, optimized_parameters, score_threshold):
    ''' Plots chi2 as a function of fitting parameters' subsets '''  
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
            im = plot_1d_error_surface(axes, score_vs_parameter_subsets[i], error_analysis_parameters[i], optimized_parameters[i], score_threshold)
        elif (dim == 2):
            im = plot_2d_error_surface(axes, score_vs_parameter_subsets[i], error_analysis_parameters[i], optimized_parameters[i], score_threshold)
        else:
            print('The score cannot be yet plotted as function of three or more parameters!')      
    left = 0
    right = float(layout[1])/float(layout[1]+1)
    bottom = 0.5 * (1-right)
    top = 1 - bottom
    fig.tight_layout(rect=[left, bottom, right, top]) 
    cax = plt.axes([right+0.05, 0.3, 0.02, 0.4])
    plt.colorbar(im, cax=cax, orientation='vertical')
    plt.text(right+0.1, 1.05, r'$\mathit{\chi^2}$', transform=cax.transAxes)
    plt.draw()
    plt.pause(0.000001)
    return fig


def keep_figures_visible():
    ''' Keep all figures visible '''
    plt.show()
    

if __name__ == '__main__':
    # Read the results of the error analysis
    filepath = get_path("Open one of the files with the results of error analysis...")
    directory = os.path.dirname(filepath) + '/'
    error_analysis_parameters, score_vs_parameter_subsets = load_error_analysis_data(directory)
    
    # Input optimized fitting parameters
    enter_optimized_parameters = False
    var = input("\nDo you want to enter the optimized parameters manually? Answer (y or n): ")
    val = str(var)
    print(val)
    if val == 'y':
        enter_optimized_parameters = True
    else:
        enter_optimized_parameters = False

    optimized_parameters = []
    if enter_optimized_parameters:
        for i in range(len(error_analysis_parameters)):
            subset_optimized_parameters = []
            for j in range(len(error_analysis_parameters[i])):
                var = input('\nEnter the optimized value of ' + \
                             error_analysis_parameters[i][j].name + ',' + \
                             str(error_analysis_parameters[i][j].spin_pair) + ',' + \
                             str(error_analysis_parameters[i][j].component) + ' : ')
                val = [float(i) for i in var.split(' ')]
                if len(val) == 1:
                    subset_optimized_parameters.append(val[0])
                else:
                    raise ValueError('More than one value obtained!')
                    sys.exit(1)  
            optimized_parameters.append(subset_optimized_parameters)
    
    # Input the score threshold
    var = input("\nEnter the chi2 threshold: ")
    val = [float(i) for i in var.split(' ')]
    if len(val) == 1:
        score_threshold = val[0]
    else:
        raise ValueError('More than one value obtained!')
        sys.exit(1)
    
    # Plot the error plot
    plot_error_surfaces(score_vs_parameter_subsets, error_analysis_parameters, optimized_parameters, score_threshold) 
    keep_figures_visible()