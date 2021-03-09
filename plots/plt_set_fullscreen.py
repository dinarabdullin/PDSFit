import os
import matplotlib
import matplotlib.pyplot as plt


def plt_set_fullscreen():
    '''
    Make a fullscreen figure
    Source: https://stackoverflow.com/questions/12439588/how-to-maximize-a-plt-show-window-using-python
    '''
    backend = str(plt.get_backend())
    mgr = plt.get_current_fig_manager()
    if backend == 'TkAgg':
        if os.name == 'nt':
            mgr.window.state('zoomed')
        else:
            mgr.resize(*mgr.window.maxsize())
    elif backend == 'wxAgg':
        mgr.frame.Maximize(True)
    elif backend == 'Qt4Agg':
        mgr.window.showMaximized()