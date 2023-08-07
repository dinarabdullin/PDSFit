def compute_number_of_data_points(experiments):
    ''' Computes the number of data ponts in the PDS time traces ''' 
    num_data_points = 0
    for experiment in experiments:
        num_data_points += experiment.s.size
    return num_data_points