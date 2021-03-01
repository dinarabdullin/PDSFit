''' Adjusts the matplotlib backend '''

import os
import matplotlib
if os.name == 'posix' and "DISPLAY" not in os.environ:
    matplotlib.use('Agg')