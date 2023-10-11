import numpy as np
import plots.set_matplotlib
from plots.set_matplotlib import best_rcparams
import matplotlib
import matplotlib.pyplot as plt
from supplement.definitions import const


markers = ["o", "s", "^", "p", "h", "*", "d", "v", "<", ">"]


def lims_and_ticks(y):
    # Limits
    ymin = np.amin(y)
    ymax = np.amax(y)
    ymin_power = np.floor(np.log10(ymin))
    ymax_power = np.floor(np.log10(ymax))
    ymin_res = 0.1 * np.floor(ymin / 10**(ymin_power - 1))
    ymax_res = 0.1 * np.ceil(ymax / 10**(ymax_power - 1))
    ymin_res = ymin_res - 0.1
    ymax_res = ymax_res
    ylim = np.array([ymin_res * 10**ymin_power, ymax_res * 10**ymax_power])
    # Ticks
    if ymin_power == ymax_power:
        if ymax_res - ymin_res < 2:
            yticks = np.arange(ymin_res, ymax_res + 0.1, 0.2) * 10**ymin_power
        else:
            yticks = np.arange(np.ceil(ymin_res), np.floor(ymax_res + 1)) * \
                10**ymin_power
    else:
        yticks = [np.ceil(ymin_res) * 10**ymin_power]
        for v in np.arange(1, np.floor(ymax_res + 1)):
            yticks.append(v * 10**ymax_power)
        yticks = np.array(yticks)
    return ylim, yticks


def plot_score(score, goodness_of_fit):
    """Plot goodness-of-fit vs. optimization step."""
    # Set x and y
    num_points = len(score)
    x = np.linspace(1, num_points, num_points, endpoint=True)
    y = score
    # Plot
    best_rcparams(1)
    fig = plt.figure(facecolor="w", edgecolor="w")
    axes = fig.gca()
    axes.semilogy(x, y, linestyle="-", color="k")
    # Formatting
    axes.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.ticklabel_format(
        style = "sci", 
        axis = "y", 
        scilimits = (0,0), 
        useMathText = True
        )
    axes.yaxis.set_minor_locator(
        matplotlib.ticker.LogLocator(
            base = 10.0, 
            subs = np.arange(1.0, 10.0) * 0.01, 
            numticks = 10
            )
        )
    axes.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes.set_xlim(0, x[-1])
    ylim, yticks = lims_and_ticks(y)
    axes.set_ylim(ylim[0], ylim[1])
    axes.set_yticks(yticks)
    plt.xlabel("Optimization step")
    plt.ylabel(const["goodness_of_fit_labels"][goodness_of_fit])
    plt.tight_layout()
    return fig


def plot_score_all_runs(score_all_runs, index_best_run, goodness_of_fit):
    """Plot goodness-of-fit vs. optimization step for several optimization runs."""
    # Set x and y
    x, y, xm, ym = [], [], [], []
    c = 0
    for r in range(len(score_all_runs)):
        score_one_run = score_all_runs[r]
        y += list(score_one_run)
        c += score_one_run.size
        xm.append(c)
        ym.append(score_one_run[-1])   
    x = np.arange(1, c + 1, 1)
    # Plot
    best_rcparams(1)
    fig = plt.figure(figsize=(12, 6), facecolor="w", edgecolor="w")
    axes = fig.gca()
    axes.semilogy(x, y, color="black")
    for i in range(len(ym)):
        if i == index_best_run:
            markerfacecolor = "red"
        else:
            markerfacecolor = "white"
        if len(ym) <= 10:
            axes.plot(xm[i], ym[i], color = "red", marker = markers[i], 
                      markerfacecolor = markerfacecolor, clip_on = False)
        else:
            axes.plot(xm[i], ym[i], color = "red", marker = "o", 
                      markerfacecolor = markerfacecolor, clip_on = False)
    # Formatting
    axes.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.ticklabel_format(
        style = "sci", 
        axis = "y", 
        scilimits = (0,0), 
        useMathText = True
        )
    axes.yaxis.set_minor_locator(
        matplotlib.ticker.LogLocator(
            base = 10.0, 
            subs = np.arange(1.0, 10.0) * 0.01, 
            numticks = 10
            )
        )
    axes.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())    
    axes.set_xlim(0, x[-1])
    ylim, yticks = lims_and_ticks(y)
    axes.set_ylim(ylim[0], ylim[1])
    axes.set_yticks(yticks)
    plt.xlabel("Optimization step")
    plt.ylabel(const["goodness_of_fit_labels"][goodness_of_fit])
    plt.tight_layout()
    return fig