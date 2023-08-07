def save_bandwidth(bandwidth, filepath):
    ''' Saves the bandwidth of detection or pump pulses'''
    file = open(filepath, 'w')
    f = bandwidth['f']
    p = bandwidth['p']
    for j in range(f.size):
        file.write('{0:<20.3f} {1:<20.6f} \n'.format(f[j], p[j]))
    file.close()