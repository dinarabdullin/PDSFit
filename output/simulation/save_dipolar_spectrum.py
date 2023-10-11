def save_dipolar_spectrum(filepath, dipolar_spectrum):
    """Save dipolar spectrum for an experimental PDS time trace and 
    a corresponding simulated PDS time trace."""
    file = open(filepath, "w")
    for i in range(dipolar_spectrum["freq"].size):
        file.write("{0:<20.3f}".format(dipolar_spectrum["freq"][i]))
        file.write("{0:<20.6f}".format(dipolar_spectrum["exp"][i]))
        file.write("{0:<20.6f}".format(dipolar_spectrum["sim"][i]))
        file.write("\n")
    file.close()