def save_time_trace(filepath, simulated_time_trace, experiment, error_bars = []):
    """Save an experimetal PDS time trace together with 
    a corresponding simulated PDS time trace."""
    file = open(filepath, "w")
    if len(error_bars) != 0:
        for i in range(experiment.t.size):
            file.write("{0:<20.3f}".format(experiment.t[i]))
            file.write("{0:<20.6f}".format(experiment.s[i]))
            file.write("{0:<20.6f}".format(simulated_time_trace[i]))
            file.write("{0:<20.6f}".format(error_bars[i][0]))
            file.write("{0:<20.6f}".format(error_bars[i][1]))
            file.write("\n")
    else:
        for i in range(experiment.t.size):
            file.write("{0:<20.3f}".format(experiment.t[i]))
            file.write("{0:<20.6f}".format(experiment.s[i]))
            file.write("{0:<20.6f}".format(simulated_time_trace[i]))
            file.write("\n")
    file.close()