import numpy as np
import plots.set_matplotlib
from plots.set_matplotlib import best_rcparams
import matplotlib
import matplotlib.pyplot as plt
from supplement.definitions import const


markers = ["o", "s", "^", "p", "h", "*", "d", "v", "<", ">"]


def plot_score_multiple_runs(score, goodness_of_fit, idx_best_solution):
    ''' Plots the score as a function of optimization step '''
    x, y, xp, yp = [], [], [], []
    count = 0
    for r in range(len(score)):
        score_one_run = score[r]
        y += list(score_one_run)
        count += score_one_run.size
        xp.append(count)
        yp.append(score_one_run[-1])   
    x = np.arange(1, count+1, 1)
    
    best_rcparams(1)
    fig = plt.figure(figsize=(12, 6), facecolor='w', edgecolor='w')
    axes = fig.gca()
    axes.semilogy(x, y, color='black')
    for i in range(len(yp)):
        if len(yp) <= 10:
            axes.plot(xp[i], yp[i], color='black', marker=markers[i], markerfacecolor='white', clip_on=False)
        else:
            axes.plot(xp[i], yp[i], color='black', marker='o', markerfacecolor='white', clip_on=False)
    
    y_minor = matplotlib.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.01, numticks=10)
    axes.yaxis.set_minor_locator(y_minor)
    axes.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes.label_outer()
    axes.set_xlim(0, x[-1])
    axes.label_outer()
    axes.set_xlim(0, x[-1])
    ymin, ymax = np.amin(y), np.amax(y)
    ymin_power, ymax_power = np.floor(np.log10(ymin)), np.floor(np.log10(ymax))
    ymin_res, ymax_res = 0.1 * np.floor(ymin / 10**(ymin_power-1)) , 0.1 * np.ceil(ymax / 10**(ymax_power-1))
    ymin_res, ymax_res = ymin_res - 0.1, ymax_res
    axes.set_ylim(ymin_res * 10**ymin_power, ymax_res * 10**ymax_power)
    if ymin_power == ymax_power:
        if ymax_res - ymin_res < 2:
            axes.set_yticks(np.arange(ymin_res, ymax_res+0.1, 0.2) * 10**ymin_power)
        else:
            axes.set_yticks(np.arange(np.ceil(ymin_res), np.floor(ymax_res+1)) * 10**ymin_power)
    else:
        yticks = [np.ceil(ymin_res) * 10**ymin_power]
        for v in np.arange(1, np.floor(ymax_res+1)):
            yticks.append(v * 10**ymax_power)
        axes.set_yticks(yticks)
    axes.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True) 
    plt.xlabel('Optimization step')
    plt.ylabel(const['goodness_of_fit_axes_labels'][goodness_of_fit])
    plt.tight_layout()
    return fig


def plot_score(score, goodness_of_fit):
    ''' Plots the score as a function of optimization step '''
    num_points = len(score)
    x = np.linspace(1, num_points, num_points, endpoint=True)
    y = score
    
    best_rcparams(1)
    fig = plt.figure(facecolor='w', edgecolor='w')
    axes = fig.gca()
    axes.semilogy(x, y, linestyle='-', color='k')
    
    y_minor = matplotlib.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.01, numticks=10)
    axes.yaxis.set_minor_locator(y_minor)
    axes.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes.label_outer()
    axes.set_xlim(0, x[-1])
    axes.label_outer()
    axes.set_xlim(0, x[-1])
    ymin, ymax = np.amin(y), np.amax(y)
    ymin_power, ymax_power = np.floor(np.log10(ymin)), np.floor(np.log10(ymax))
    ymin_res, ymax_res = 0.1 * np.floor(ymin / 10**(ymin_power-1)) , 0.1 * np.ceil(ymax / 10**(ymax_power-1))
    ymin_res, ymax_res = ymin_res - 0.1, ymax_res
    axes.set_ylim(ymin_res * 10**ymin_power, ymax_res * 10**ymax_power)
    if ymin_power == ymax_power:
        if ymax_res - ymin_res < 2:
            axes.set_yticks(np.arange(ymin_res, ymax_res+0.1, 0.2) * 10**ymin_power)
        else:
            axes.set_yticks(np.arange(np.ceil(ymin_res), np.floor(ymax_res+1)) * 10**ymin_power)
    else:
        yticks = [np.ceil(ymin_res) * 10**ymin_power]
        for v in np.arange(1, np.floor(ymax_res+1)):
            yticks.append(v * 10**ymax_power)
        axes.set_yticks(yticks)
    axes.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True) 
    plt.xlabel('Optimization step')
    plt.ylabel(const['goodness_of_fit_axes_labels'][goodness_of_fit])
    plt.tight_layout()
    return fig


def update_score_plot(fig, score, goodness_of_fit):
    ''' Re-plots the score as a function of optimization step '''
    num_points = len(score)
    x = np.linspace(1,num_points,num_points)
    y = score
    
    y_minor = matplotlib.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.01, numticks=10)
    axes.yaxis.set_minor_locator(y_minor)
    axes.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes.label_outer()
    axes.set_xlim(0, x[-1])
    axes.label_outer()
    axes.set_xlim(0, x[-1])
    ymin, ymax = np.amin(y), np.amax(y)
    ymin_power, ymax_power = np.floor(np.log10(ymin)), np.floor(np.log10(ymax))
    ymin_res, ymax_res = 0.1 * np.floor(ymin / 10**(ymin_power-1)) , 0.1 * np.ceil(ymax / 10**(ymax_power-1))
    ymin_res, ymax_res = ymin_res - 0.1, ymax_res
    axes.set_ylim(ymin_res * 10**ymin_power, ymax_res * 10**ymax_power)
    if ymin_power == ymax_power:
        if ymax_res - ymin_res < 2:
            axes.set_yticks(np.arange(ymin_res, ymax_res+0.1, 0.2) * 10**ymin_power)
        else:
            axes.set_yticks(np.arange(np.ceil(ymin_res), np.floor(ymax_res+1)) * 10**ymin_power)
    else:
        yticks = [np.ceil(ymin_res) * 10**ymin_power]
        for v in np.arange(1, np.floor(ymax_res+1)):
            yticks.append(v * 10**ymax_power)
        axes.set_yticks(yticks)
    axes.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True) 
    plt.xlabel('Optimization step')
    plt.ylabel(const['goodness_of_fit_axes_labels'][goodness_of_fit])
    plt.tight_layout()


def close_score_plot(fig):
    ''' Closes the score plot '''
    plt.close(fig)