''' Adjusts matplotlib's backend and rcParams '''
import os
import matplotlib
if os.name == 'posix' and "DISPLAY" not in os.environ:
    matplotlib.use('Agg')
from matplotlib import rcParams
rcParams['xtick.direction'] = 'out'
rcParams['ytick.direction'] = 'out'
rcParams['axes.facecolor']= 'white'
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = 'Arial'

def best_rcparams(n):
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