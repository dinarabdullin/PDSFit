from supplement.definitions import const


def save_goodness_of_fit(goodness_of_fit, directory):
    ''' Saves the goodness-of-fit as a function of optimization step '''
    filepath = directory + 'goodness_of_fit.dat'
    file = open(filepath, 'w')
    file.write("{:<15}{:<15}\n".format('No. iteration', 'Chi2'))
    for i in range(goodness_of_fit.size):
        file.write('{0:<15d} {1:<15.6f} \n'.format(i+1, goodness_of_fit[i]))
    file.close()


def save_fitting_parameters(parameters_indices, optimized_parameters_values, fixed_parameters_values, optimized_parameters_errors, directory):    
    ''' Saves optimized and fixed fitting parameters ''' 
    filepath = directory + 'fitting_parameters.dat'
    file = open(filepath, 'w')
    file.write("{:<20}{:<15}{:<15}{:<15}{:<15}{:<15}\n".format('Parameter', 'No. spin pair', 'No. component', 'Optimized', 'Value', 'Precision'))
    for variable_name in const['fitting_parameters_names']:
        variable_indices = parameters_indices[variable_name]
        for i in range(len(variable_indices)):
            for j in range(len(variable_indices[i])):
                file.write('{:<20}'.format(const['fitting_parameters_names_and_units'][variable_name]))
                file.write('{:<15}'.format(i+1))
                file.write('{:<15}'.format(j+1))
                variable_id = variable_indices[i][j]
                if variable_id.opt:
                    file.write('{:<15}'.format('yes'))
                else:
                    file.write('{:<15}'.format('no'))
                if variable_id.opt:
                    variable_value = optimized_parameters_values[variable_id.idx] / const['fitting_parameters_scales'][variable_name]
                    file.write('{:<15.4}'.format(variable_value))
                else:
                    variable_value = fixed_parameters_values[variable_id.idx]  / const['fitting_parameters_scales'][variable_name]
                    file.write('{:<15.4}'.format(variable_value))
                if variable_id.opt:
                    if optimized_parameters_errors != []:
                        if optimized_parameters_errors[variable_id.idx] != 0:
                            variable_error = optimized_parameters_errors[variable_id.idx] / const['fitting_parameters_scales'][variable_name]
                            file.write('{:<15.4}'.format(variable_error))
                        else:
                            file.write('{:<15}'.format('nan'))
                    else:
                        file.write('{:<15}'.format('nan'))
                else:
                    file.write('{:<15}'.format('nan'))    
                file.write('\n') 
    file.close()


def save_fits(simulated_time_traces, experiments, directory):
    ''' Saved fits to the experimental PDS time traces '''
    for i in range(len(experiments)):
        filepath = directory + 'fit_' + experiments[i].name + ".dat"
        file = open(filepath, 'w')
        file.write("{:<15}{:<15}{:<15}\n".format('t', 'exp', 'fit'))    
        t = experiments[i].t
        s_exp = experiments[i].s
        s_sim = simulated_time_traces[i]['s']
        for j in range(t.size):
            file.write('{0:<15.7f} {1:<15.7f} {2:<15.7f} \n'.format(t[j], s_exp[j], s_sim[j]))
        file.close()
        

def save_fitting_output(goodness_of_fit, optimized_parameters_values, optimized_parameters_errors, fitting_parameters, simulated_time_traces, experiments, directory):
    ''' Saves the fitting output '''
    # Save the goodness-of-fit (vs optimization step)
    save_goodness_of_fit(goodness_of_fit, directory)
    # Save the fitting parameters
    save_fitting_parameters(fitting_parameters['indices'], optimized_parameters_values, fitting_parameters['values'], optimized_parameters_errors, directory)
    # Save the fits to the experimental PDS time traces
    save_fits(simulated_time_traces, experiments, directory)