def save_simulated_time_trace(simulated_time_trace, error_bars_simulated_time_trace, experiment, filepath):
    ''' Saves a simulated PDS time trace '''
    file = open(filepath, 'w')
    if experiment.s != []:
        if error_bars_simulated_time_trace != []:
            for j in range(simulated_time_trace['t'].size):
                file.write('{0:<20.3f}{1:<20.6f}{2:<20.6f}{3:<20.6f}{4:<20.6f}{5:<20.6f}\n'.format(simulated_time_trace['t'][j], 
                                                                                                   experiment.s[j], 
                                                                                                   experiment.s_im[j],
                                                                                                   simulated_time_trace['s'][j],
                                                                                                   error_bars_simulated_time_trace[j][0],
                                                                                                   error_bars_simulated_time_trace[j][1]))
        else:
            for j in range(simulated_time_trace['t'].size):
                file.write('{0:<20.3f}{1:<20.6f}{2:<20.6f}{3:<20.6f}\n'.format(simulated_time_trace['t'][j], 
                                                                               experiment.s[j], 
                                                                               experiment.s_im[j],
                                                                               simulated_time_trace['s'][j]))   
    else:
        if error_bars_simulated_time_trace != []:
            for j in range(simulated_time_trace['t'].size):
                file.write('{0:<20.3f}{1:<20.6f}{2:<20.6f}{3:<20.6f}\n'.format(simulated_time_trace['t'][j],
                                                                               simulated_time_trace['s'][j],
                                                                               error_bars_simulated_time_trace[j][0],
                                                                               error_bars_simulated_time_trace[j][1]))
        else:
            for j in range(simulated_time_trace['t'].size):
                file.write('{0:<20.3f}{1:<20.6f}\n'.format(simulated_time_trace['t'][j],
                                                           simulated_time_trace['s'][j]))
    file.close()