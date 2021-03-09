def save_bandwidths(bandwidths, experiments, directory):
    ''' Save the bandwidths of detection and pump pulses'''
    for i in range(len(experiments)):
        for key in bandwidths[i]:
            filepath = directory + key + '_' + experiments[i].name + ".dat"
            file = open(filepath, 'w')
            f = bandwidths[i][key]['f']
            p = bandwidths[i][key]['p']
            for j in range(f.size):
                file.write('{0:<12.7f} {1:<12.7f} \n'.format(f[j], p[j]))
            file.close()