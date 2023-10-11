def save_background(filepath, background, experiment, error_bars = []):
    """Save an experimental PDS time trace together with its simulated background."""
    file = open(filepath, "w")
    if len(error_bars) != 0:
        for i in range(experiment.t.size):
            file.write("{0:<20.3f}".format(experiment.t[i]))
            file.write("{0:<20.6f}".format(experiment.s[i]))
            file.write("{0:<20.6f}".format(background[i]))
            file.write("{0:<20.6f}".format(error_bars[0][i]))
            file.write("{0:<20.6f}".format(error_bars[1][i]))
            file.write("\n")
    else:
        for i in range(experiment.t.size):
            file.write("{0:<20.3f}".format(experiment.t[i]))
            file.write("{0:<20.6f}".format(experiment.s[i]))
            file.write("{0:<20.6f}".format(background[i]))
            file.write("\n")
    file.close()