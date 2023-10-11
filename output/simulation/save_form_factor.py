def save_form_factor(filepath, form_factor, experiment):
    """Save form factors for an experimental PDS time trace and 
    a corresponding simulated PDS time trace."""
    file = open(filepath, "w")
    for i in range(experiment.t.size):
        file.write("{0:<20.3f}".format(experiment.t[i]))
        file.write("{0:<20.6f}".format(form_factor["exp"][i]))
        file.write("{0:<20.6f}".format(form_factor["sim"][i]))
        file.write("\n")
    file.close()