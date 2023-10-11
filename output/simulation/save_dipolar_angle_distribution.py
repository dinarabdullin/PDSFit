def save_dipolar_angle_distribution(filepath, dipolar_angle_distribution):
    """Save a simulated distribution of the dipolar angle."""
    file = open(filepath, "w")
    for i in range(dipolar_angle_distribution["angle"].size):
        file.write("{0:<20.0f}".format(dipolar_angle_distribution["angle"][i]))
        file.write("{0:<20.6f}".format(dipolar_angle_distribution["prob"][i]))
        file.write("\n")
    file.close()