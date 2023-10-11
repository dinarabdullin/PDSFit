def save_epr_spectrum(filepath, epr_spectrum):
    """Save the simulated EPR spectrum of a spin system."""
    file = open(filepath, "w")
    for i in range(epr_spectrum["freq"].size):
        file.write("{0:<20.3f}".format(epr_spectrum["freq"][i]))
        file.write("{0:<20.6f}".format(epr_spectrum["prob"][i]))
        file.write("\n")
    file.close()