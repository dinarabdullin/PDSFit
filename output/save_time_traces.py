def save_time_traces(simulated_time_traces, experiments, directory):
    ''' Save simulated PDS time traces '''
    for i in range(len(experiments)):
        filepath = directory + 'time_trace_' + experiments[i].name + ".dat"
        file = open(filepath, 'w')
        t = experiments[i].t
        s_exp = experiments[i].s
        s_sim = simulated_time_traces[i]['s']
        for j in range(t.size):
            file.write('{0:<12.7f} {1:<12.7f} {2:<12.7f} \n'.format(t[j], s_exp[j], s_sim[j]))
        file.close()