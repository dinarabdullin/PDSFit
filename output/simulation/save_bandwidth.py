def save_bandwidth(filepath, bandwidth):
    """Save the bandwidth of detection or pump pulses."""
    file = open(filepath, 'w')
    for i in range(bandwidth["freq"].size):
        file.write("{0:<20.3f}".format(bandwidth["freq"][i]))
        file.write("{0:<20.6f}".format(bandwidth["prob"][i]))
        file.write("\n")
    file.close()