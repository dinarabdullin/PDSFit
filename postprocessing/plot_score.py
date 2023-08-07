import os
import io
import wx
import numpy as np
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['xtick.direction'] = 'out'
rcParams['ytick.direction'] = 'out'
rcParams['axes.facecolor']= 'white'
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = 'Arial'
rcParams['lines.linewidth'] = 1
rcParams['xtick.major.size'] = 4
rcParams['xtick.major.width'] = 1
rcParams['ytick.major.size'] = 4
rcParams['ytick.major.width'] = 1
rcParams['font.size'] = 18
rcParams['lines.markersize'] = 12
rcParams['lines.markeredgewidth'] = 1

ystep = 0
reverse = False
markers = ["s", "o", "^", "p", "h", "*", "d", "v", "<", ">"]


def get_filepath(message):
    app = wx.App(None) 
    dialog = wx.FileDialog(None, message, wildcard='*.*', style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
    if dialog.ShowModal() == wx.ID_OK:
        filepath = dialog.GetPath()
    else:
        filepath = ''
    return filepath


def load_score(directory):
    # Find all error analysis files
    filenames = []
    c = 1
    while True:
        filename = directory + 'score_run' + str(c) + '.dat'
        c += 1
        if os.path.exists(filename):
            filenames.append(filename)
        else:
            break
    # Set the error analysis parameters
    score_all_runs = []
    x_offset = 0
    x_merged, y_merged = [], []
    for filename in filenames:
        x, y = [], []
        file = open(filename, 'r')
        for line in file:
            data = line.split()
            x.append(float(data[0]) + x_offset)
            y.append(float(data[1]))
            x_merged.append(float(data[0]) + x_offset)
            y_merged.append(float(data[1]))
        x = np.array(x)
        y = np.array(y)
        score_all_runs.append([x, y])
        file.close()
        x_offset += x.size - 1
    score_merged = [x_merged, y_merged]
    return score_all_runs, score_merged


if __name__ == '__main__':
    
    # Load the score values
    filepath = get_filepath("Open the file with score values...")
    directory = os.path.dirname(filepath) + '/'
    score_all_runs, score_merged = load_score(directory)
    
    fig = plt.figure(figsize=(12,6), facecolor='w', edgecolor='w')
    axes = fig.gca()   
    x, y = score_merged[0], score_merged[1]
    axes.semilogy(x, y, color='black', linewidth=1.5)
    #axes.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(numticks=999, subs=()))
    y_minor = matplotlib.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.01, numticks=10)
    axes.yaxis.set_minor_locator(y_minor)
    axes.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    for i in range(len(score_all_runs)):
        xs, ys = score_all_runs[i][0], score_all_runs[i][1]
        xp, yp = xs[-1], ys[-1]
        axes.plot(xp, yp, color='red', marker=markers[i], markerfacecolor='white', clip_on=False)
    axes.label_outer()
    axes.set_xlim(0, x[-1])
    ymin, ymax = np.amin(y), np.amax(y)
    ymin_power, ymax_power = np.floor(np.log10(ymin)), np.floor(np.log10(ymax))
    ymin_res, ymax_res = 0.1 * np.floor(ymin / 10**(ymin_power-1)) , 0.1 * np.ceil(ymax / 10**(ymax_power-1))
    ymin_res, ymax_res = ymin_res - 0.1, ymax_res
    print([ymin_res, ymax_res])
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
    plt.ylabel(r'$\mathit{\chi^2}$')
    fig.tight_layout() 
    labels = [item.get_text() for item in axes.get_yticklabels()]
    filename_figure = directory + 'score_all_runs_edited.png'
    plt.savefig(filename_figure, format='png', dpi=600)
    plt.draw()
    plt.pause(0.000001)
    plt.show()