def save_simulated_time_trace(simulated_time_trace, experimental_time_trace, filepath):
    ''' Saves a simulated PDS time trace '''
    file = open(filepath, 'w')
    t = simulated_time_trace['s']
    s_sim = simulated_time_trace['s']
    s_exp = experimental_time_trace
    if s_exp != []:
        for j in range(t.size):
            file.write('{0:<15.7f} {1:<15.7f} {2:<15.7f} \n'.format(t[j], s_exp[j], s_sim[j]))
    else:
        for j in range(t.size):
            file.write('{0:<15.7f} {1:<15.7f} \n'.format(t[j], s_sim[j]))
    file.close()