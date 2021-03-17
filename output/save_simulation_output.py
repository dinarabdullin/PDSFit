import os
import sys


def save_epr_spectrum(spectrum, directory, experiment_name):
    ''' Saves a simulated EPR spectrum '''
    filepath = directory + 'epr_spectrum_' + experiment_name + '.dat'
    file = open(filepath, 'w')
    for i in range(spectrum['f'].size):
        file.write('{0:<15.4f} {1:<15.4f} \n'.format(spectrum['f'][i], spectrum['p'][i]))
    file.close()


def save_bandwidths(bandwidths, experiments, directory):
    ''' Saves the bandwidths of detection and pump pulses'''
    for i in range(len(experiments)):
        for key in bandwidths[i]:
            filepath = directory + key + '_' + experiments[i].name + ".dat"
            file = open(filepath, 'w')
            f = bandwidths[i][key]['f']
            p = bandwidths[i][key]['p']
            for j in range(f.size):
                file.write('{0:<15.7f} {1:<15.7f} \n'.format(f[j], p[j]))
            file.close()


def save_time_traces(simulated_time_traces, experiments, directory):
    ''' Saves simulated PDS time traces '''
    for i in range(len(experiments)):
        filepath = directory + 'time_trace_' + experiments[i].name + ".dat"
        file = open(filepath, 'w')
        t = experiments[i].t
        s_exp = experiments[i].s
        s_sim = simulated_time_traces[i]['s']
        for j in range(t.size):
            file.write('{0:<15.7f} {1:<15.7f} {2:<15.7f} \n'.format(t[j], s_exp[j], s_sim[j]))
        file.close()


def save_simulation_output(epr_spectra, bandwidths, simulated_time_traces, experiments, directory):
    ''' Saves the simulation output '''
    # Save the EPR spectrum of the spin system
    save_epr_spectrum(epr_spectra[0], directory, experiments[0].name)
    # Save the bandwidths
    save_bandwidths(bandwidths, experiments, directory)
    # Save the time traces
    save_time_traces(simulated_time_traces, experiments, directory)