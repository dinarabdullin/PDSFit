def save_simulated_spectrum(save_simulated_spectrum, filepath):
    ''' Saves a simulated dipolar spectrum '''
    file = open(filepath, 'w')
    f = save_simulated_spectrum['f']
    p_exp = save_simulated_spectrum['pe']
    p_sim = save_simulated_spectrum['p']
    if p_exp != []:
        for j in range(f.size):
            file.write('{0:<20.3f} {1:<20.6f} {2:<20.6f}\n'.format(f[j], p_exp[j], p_sim[j]))
    else:
        for j in range(f.size):
            file.write('{0:<20.3f} {1:<20.6f} \n'.format(f[j], p_sim[j]))
    file.close()