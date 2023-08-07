def save_background_time_trace(background_time_trace, error_bars_background_time_trace, experiment, filepath):
    ''' Saves a PDS time trace with its background fit '''
    file = open(filepath, 'w')
    if experiment.s != []:
        if error_bars_background_time_trace != []:
            for j in range(background_time_trace['t'].size):
                file.write('{0:<20.3f}{1:<20.6f}{2:<20.6f}{3:<20.6f}{4:<20.6f}{5:<20.6f}\n'.format(background_time_trace['t'][j],
                                                                                                   experiment.s[j],
                                                                                                   experiment.s_im[j],
                                                                                                   background_time_trace['s'][j],
                                                                                                   error_bars_background_time_trace[j][0],
                                                                                                   error_bars_background_time_trace[j][1]))
        else:
            for j in range(background_time_trace['t'].size):
                file.write('{0:<20.3f}{1:<20.6f}{2:<20.6f}{3:<20.6f}\n'.format(background_time_trace['t'][j],
                                                                               experiment.s[j],
                                                                               experiment.s_im[j],
                                                                               background_time_trace['s'][j]))
    else:
        if error_bars_background_time_trace != []:
            for j in range(background_time_trace['t'].size):
                file.write('{0:<20.3f}{1:<20.6f}{2:<20.6f}{3:<20.6f}\n'.format(background_time_trace['t'][j],
                                                                               background_time_trace['s'][j],
                                                                               error_bars_background_time_trace[j][0],
                                                                               error_bars_background_time_trace[j][1]))
        else:
            for j in range(background_time_trace['t'].size):
                file.write('{0:<20.3f}{1:<20.6f}\n'.format(background_time_trace['t'][j],
                                                           background_time_trace['s'][j]))
    file.close()