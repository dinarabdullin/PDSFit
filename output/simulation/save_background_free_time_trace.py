def save_background_free_time_trace(background_free_time_trace, filepath):
    ''' Saves the background-free part of a PDS time trace '''
    file = open(filepath, 'w')
    t = background_free_time_trace['t']
    s_exp = background_free_time_trace['se']
    s_sim = background_free_time_trace['s']
    if s_exp != []:
        for j in range(t.size):
            file.write('{0:<20.7f} {1:<20.7f} {2:<20.7f}\n'.format(t[j], s_exp[j], s_sim[j]))
    else:
        for j in range(t.size):
            file.write('{0:<20.7f} {1:<20.7f} \n'.format(t[j], s_sim[j]))
    file.close()