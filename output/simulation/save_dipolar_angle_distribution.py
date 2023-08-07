def save_dipolar_angle_distribution(dipolar_angle_distribution, filepath):
    ''' Saves the simulated distribution of dipolar angles '''
    file = open(filepath, 'w')
    v = dipolar_angle_distribution['v']
    p = dipolar_angle_distribution['p']
    for i in range(v.size):
        file.write('{0:<20.0f} {1:<20.6f}\n'.format(v[i], p[i]))
    file.close()