
def save_timetrace(timetrace, directory):
    filepath = directory + "timetrace.dat"
    file = open(filepath, 'w')
    for i in range(timetrace['t'].size):
        file.write('{0:<12.4f} {1:<12.4f} \n'.format(timetrace['t'][i], timetrace['s'][i]))
    file.close()