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
    """Set matplotlib.rcParams in dependence of the subplots number."""
    if   n == 1:
        rcParams['lines.linewidth'] = 2
        rcParams['xtick.major.size'] = 8
        rcParams['xtick.major.width'] = 1.5
        rcParams['ytick.major.size'] = 8
        rcParams['ytick.major.width'] = 1.5
        rcParams['font.size'] = 18
        rcParams['lines.markersize'] = 10
        rcParams['lines.markeredgewidth'] = 0.5
    elif n >= 2 and n <= 4:
        rcParams['lines.linewidth'] = 1.5
        rcParams['xtick.major.size'] = 4
        rcParams['xtick.major.width'] = 1.5
        rcParams['ytick.major.size'] = 4
        rcParams['ytick.major.width'] = 1
        rcParams['font.size'] = 12
        rcParams['lines.markersize'] = 8
        rcParams['lines.markeredgewidth'] = 0.5
    elif n >= 4 and n < 10:
        rcParams['lines.linewidth'] = 1
        rcParams['xtick.major.size'] = 4
        rcParams['xtick.major.width'] = 1
        rcParams['ytick.major.size'] = 4
        rcParams['ytick.major.width'] = 1
        rcParams['font.size'] = 12
        #rcParams['lines.markersize'] = 6
        #rcParams['lines.markeredgewidth'] = 1
        rcParams['lines.markersize'] = 8
        rcParams['lines.markeredgewidth'] = 0.5
    elif n >= 10 and n < 13:
        rcParams['lines.linewidth'] = 1
        rcParams['xtick.major.size'] = 4
        rcParams['xtick.major.width'] = 1
        rcParams['ytick.major.size'] = 4
        rcParams['ytick.major.width'] = 1
        rcParams['font.size'] = 11.5
        rcParams['lines.markersize'] = 6
        rcParams['lines.markeredgewidth'] = 0.5
    elif n >= 13:
        rcParams['axes.linewidth'] = 0.5
        rcParams['lines.linewidth'] = 0.5
        rcParams['xtick.major.size'] = 3
        rcParams['xtick.major.width'] = 0.5
        rcParams['ytick.major.size'] = 3
        rcParams['ytick.major.width'] = 0.5
        rcParams['font.size'] = 7.5
        rcParams['lines.markersize'] = 5
        rcParams['lines.markeredgewidth'] = 0.5