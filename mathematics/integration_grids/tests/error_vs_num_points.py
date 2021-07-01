import math
import matplotlib.pyplot as plt


def error_vs_num_points(grid, integrand, expected_integral_value, num_points_array, integration_range):
    error_vs_num_points = {}
    for i in num_points_array:
        if integration_range == []:
            integral_value = grid.integrate_function(integrand, i)
        else:
            integral_value = grid.integrate_function(integrand, i, integration_range[0], integration_range[1])
        error = abs(100*(integral_value-expected_integral_value)/expected_integral_value)
        error_vs_num_points[i] = error
    return error_vs_num_points

def plot_error_vs_num_points(data, labels, titel, filename):
    fig = plt.figure()
    axes = fig.gca()
    for i in range(len(labels)):
        axes.plot(data[i].keys(), data[i].values(), marker =".", label = labels[i])
    axes.set_ylabel('Relative error in %')
    axes.set_xlabel('Number of grid points')
    axes.set_ylim([-5,100])
    plt.xscale('log',base=10) 
    axes.legend(prop={'size': 8})
    plt.title(titel, fontsize = 10)
    fig.savefig("mathematics/integration_grids/tests/"+filename+".png", format='png', dpi=600)
    plt.show()