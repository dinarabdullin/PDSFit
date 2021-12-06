def save_background(background_time_trace, experimental_time_trace, experimental_time_trace_im, filepath):
    ''' Saves the PDS time trace with its background fit '''
    file = open(filepath, 'w')
    t = background_time_trace['t']
    s_bckg = background_time_trace['s']
    s_exp = experimental_time_trace
    s_exp_im = experimental_time_trace_im
    if s_exp != []:
        for j in range(t.size):
            file.write('{0:<20.7f} {1:<20.7f} {2:<20.7f} {3:<20.7f}\n'.format(t[j], s_exp[j], s_bckg[j], s_exp_im[j]))
    else:
        for j in range(t.size):
            file.write('{0:<20.7f} {1:<20.7f} \n'.format(t[j], s_bckg[j]))
    file.close()