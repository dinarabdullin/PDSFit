'''
Save a bandwidth profile
'''

def save_bandwidth(bandwidth, directory, filename):
    filepath = directory + filename + ".dat"
    file = open(filepath, 'w')
    for i in range(bandwidth["f"].size):
        file.write('{0:<12.4f} {1:<12.4f} \n'.format(bandwidth["f"][i], bandwidth["p"][i]))
    file.close()