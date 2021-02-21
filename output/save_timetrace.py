
def save_timetrace(t, s, directory):
    filepath = directory + "timetrace.dat"
    file = open(filepath, 'w')
    for i in range(t.size):
        file.write('{0:<12.4f} {1:<12.4f} \n'.format(t[i], s[i]))
    file.close()