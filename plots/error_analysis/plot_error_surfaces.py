import copy
import numpy as np
import plots.set_matplotlib
from plots.set_matplotlib import best_rcparams
import matplotlib.pyplot as plt
from plots.best_layout import best_layout
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
from mathematics.find_nearest import find_nearest
from supplement.definitions import const


markers = ["o", "s", "^", "p", "h", "*", "d", "v", "<", ">"]


def plot_error_surfaces(
    error_surfaces, chi2_minimum, chi2_thresholds, optimized_model_parameters, 
    fitting_parameters, show_uncertainty_interval = False
    ):
    """Plot error surfaces."""
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


def plot_error_surface_1d(
    axes, error_surface, y_min, y_thr, p_opt, multimodal_distributions = False, show_uncertainty_interval = False,
    ):
    """Plot a one-dimensional error surface."""
    parameter, xv, yv = error_surface["par"][0], error_surface["x"][0], error_surface["y"]
    xv = xv / const["model_parameter_scales"][parameter.name]
    x_min, x_max = parameter.get_range()[0], parameter.get_range()[1]
    x_min = x_min / const["model_parameter_scales"][parameter.name]
    x_max = x_max / const["model_parameter_scales"][parameter.name]
    n_opt = len(p_opt)
    xv_opt, yv_opt = [], []
    for i in range(n_opt):
        x_opt = p_opt[i][parameter.get_index()]
        x_opt = x_opt / const["model_parameter_scales"][parameter.name]
        index_opt = find_nearest(xv, x_opt)
        x_opt, y_opt = xv[index_opt], yv[index_opt]
        xv_opt.append(x_opt)
        yv_opt.append(y_opt)
    # Plot a one-dimensional error surface
    im = axes.scatter(xv, yv, c = yv, cmap = "jet_r", vmin = y_min + y_thr, vmax = 1.5 * y_min + y_thr)
    axes.set_xlim(x_min, x_max)
    axes.set_xticks(np.linspace(x_min, x_max, 3))
    if multimodal_distributions:
        x_label = \
            const["model_parameter_labels"][parameter.name][0] + \
            r"$_{%d}$" % (parameter.component + 1) + " " + \
            const["model_parameter_labels"][parameter.name][1]
    else:
        x_label = \
            const["model_parameter_labels"][parameter.name][0] + " " + \
            const["model_parameter_labels"][parameter.name][1]
    y_label = r"$\mathit{\chi^2}$"
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    # Depict the optimized value of a fitting parameter
    for i in range(n_opt):
        if n_opt <= 10:
            axes.plot(
                xv_opt[i], yv_opt[i], color = "black", marker = markers[i], markerfacecolor = "white", clip_on = False
                )
        else:
            axes.plot(
                xv_opt[i], yv_opt[i], color = "black", marker = "o", markerfacecolor = "white", clip_on = False
                )
    axes.ticklabel_format(axis = "y", style = "sci", scilimits = (0,0), useMathText = True) 
    # Depict an uncertainty interval as a gray shade
    if show_uncertainty_interval:
        indices_uncertainty_interval = np.where(yv <= y_min + y_thr)[0]
        if len(indices_uncertainty_interval) >= 2:
            xv_ui = xv[indices_uncertainty_interval]
            lb_ui, ub_ui = np.amin(xv_ui), np.amax(xv_ui)
            if ub_ui > lb_ui:
                axes.axvspan(lb_ui, ub_ui, facecolor="lightgray", alpha=0.3, label="confidence\n interval")
                axes.plot(
                    xv, (y_min + y_thr) * np.ones(xv.size), 'k--', label = r'$\mathit{\chi^{2}_{min}}$ + $\mathit{\Delta\chi^{2}}$'
                    )
    # Make axes square
    xl, xh = axes.get_xlim()
    yl, yh = axes.get_ylim()
    axes.set_aspect((xh - xl) / (yh - yl))
    return im


def plot_error_surface_2d(
    axes, error_surface, y_min, y_thr, p_opt, multimodal_distributions = False
    ):
    """Plot a two-dimensional error surface."""
    parameters, xv, yv = error_surface["par"], error_surface["x"], error_surface["y"]
    parameter1, parameter2 = parameters[0], parameters[1]
    xg = np.reshape(xv, [2] + [int(np.sqrt(yv.shape))] * 2)
    yg = np.reshape(yv, [int(np.sqrt(yv.shape))] * 2)
    xg[0] = xg[0] / const["model_parameter_scales"][parameter1.name]
    xg[1] = xg[1] / const["model_parameter_scales"][parameter2.name]
    x1_min, x1_max = parameter1.get_range()[0], parameter1.get_range()[1]
    x2_min, x2_max = parameter2.get_range()[0], parameter2.get_range()[1]
    x1_min = x1_min / const["model_parameter_scales"][parameter1.name]
    x1_max = x1_max / const["model_parameter_scales"][parameter1.name]
    x2_min = x2_min / const["model_parameter_scales"][parameter2.name]
    x2_max = x2_max / const["model_parameter_scales"][parameter2.name]
    n_opt = len(p_opt)
    xv_opt = []
    for i in range(n_opt):
        x1_opt = p_opt[i][parameter1.get_index()] 
        x2_opt = p_opt[i][parameter2.get_index()]
        x1_opt = x1_opt / const["model_parameter_scales"][parameter1.name]
        x2_opt = x2_opt / const["model_parameter_scales"][parameter2.name]
        xv_opt.append([x1_opt, x2_opt])
    # Plot a two-dimensional error surface
    im = axes.pcolor(xg[0], xg[1], yg, cmap = "jet_r", vmin = y_min + y_thr, vmax = 1.5 * y_min + y_thr)
    axes.set_xlim(x1_min, x1_max)
    axes.set_xticks(np.linspace(x1_min, x1_max, 3))
    axes.set_ylim(x2_min, x2_max)
    axes.set_yticks(np.linspace(x2_min, x2_max, 3))
    if multimodal_distributions:    
        x_label = \
            const["model_parameter_labels"][parameter1.name][0] + \
            r"$_{%d}$" % (parameter1.component + 1) + " " + \
            const["model_parameter_labels"][parameter1.name][1]
        y_label = \
            const["model_parameter_labels"][parameter2.name][0] + \
            r"$_{%d}$" % (parameter2.component + 1) + " " + \
            const["model_parameter_labels"][parameter2.name][1]
    else:
        x_label = \
            const["model_parameter_labels"][parameter1.name][0] + " " + \
            const["model_parameter_labels"][parameter1.name][1]
        y_label = \
            const["model_parameter_labels"][parameter2.name][0] + " " + \
            const["model_parameter_labels"][parameter2.name][1]
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    # Depict the optimized values of fitting parameters
    for i in range(n_opt):
        if n_opt <= 10:
            axes.plot(
                xv_opt[i][0], xv_opt[i][1], color = "black", marker = markers[i], markerfacecolor = "white", clip_on = False
                )
        else:
            axes.plot(
                xv_opt[i][0], xv_opt[i][1], color = "black", marker = "o", markerfacecolor = "white", clip_on = False
                )
    # Make axes square
    xl, xh = axes.get_xlim()
    yl, yh = axes.get_ylim()
    axes.set_aspect((xh - xl) / (yh - yl))
    return im