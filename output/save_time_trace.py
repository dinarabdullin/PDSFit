''' Save a simulated PDS time trace '''

def save_time_trace(t, s_sim, s_exp, directory, filename='timetrace.dat'):
    filepath = directory + filename
    file = open(filepath, 'w')
    for i in range(t.size):
        file.write('{0:<12.7f} {1:<12.7f} {0:<12.7f} \n'.format(t[i], s_exp[i], s_sim[i]))
    file.close()