'''
Create an output directory
'''

import os
import errno
import datetime
import shutil

def make_output_directory(output_settings, filepath_config):
    if output_settings['save_data']:
        parent_directory = output_settings['directory']
        config_directory, config_name = os.path.split(os.path.abspath(filepath_config))
        if (parent_directory):
            output_directory = parent_directory
        else:
            output_directory = config_directory
        now = datetime.datetime.now()
        folder = now.strftime("%Y-%m-%d_%H-%M")
        output_directory = output_directory + "/" + folder + "/"
        try:
            os.makedirs(output_directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        shutil.copy2(filepath_config, output_directory + config_name)
        output_settings['directory'] = output_directory