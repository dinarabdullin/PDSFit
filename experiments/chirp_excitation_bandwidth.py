import numpy as np


def chirp_excitation_bandwidth(frequency_center, frequency_sweep_width, pulse_length, rise_time, critical_adiabaticity):
    
    #print(maximal_rabi_frequency)
    frequency_step = 0.01
    time_step = 0.1
    frequency_axis = np.arange(frequency_center - 0.5*frequency_sweep_width, frequency_center + 0.5*frequency_sweep_width + frequency_step, frequency_step)
    time_axis = np.arange(0, pulse_length + time_step, time_step)
    frequency_axis_size = frequency_axis.size
    time_axis_size = time_axis.size
    
    maximal_rabi_frequency = np.sqrt(critical_adiabaticity * frequency_sweep_width / pulse_length)
    print("Maximal Rabi frequency : %f GHz" % maximal_rabi_frequency)
    microwave_frequencies = frequency_center - 0.5*frequency_sweep_width + frequency_sweep_width * time_axis / pulse_length
    rabi_frequencies = np.zeros(time_axis_size)
    adiabaticity_array = np.zeros((frequency_axis_size, time_axis_size))
    
    for i in range(frequency_axis_size):
        for j in range(time_axis_size):
            
            if rise_time == 0:
                rabi_frequency = maximal_rabi_frequency
                rabi_frequency_derivative = 0
            else:
                if time_axis[j] < rise_time:
                    rabi_frequency = maximal_rabi_frequency * np.sin(0.5*np.pi * time_axis[j] / rise_time)
                    rabi_frequency_derivative = maximal_rabi_frequency * (0.5*np.pi / rise_time) * np.cos(0.5*np.pi * time_axis[j] / rise_time)
                elif time_axis[j] > pulse_length - rise_time:
                    rabi_frequency = maximal_rabi_frequency * np.sin(0.5*np.pi * (pulse_length - time_axis[j])/rise_time)
                    rabi_frequency_derivative = maximal_rabi_frequency * (-0.5*np.pi / rise_time) * np.cos(0.5*np.pi * (pulse_length - time_axis[j]) / rise_time)
                else:
                    rabi_frequency = maximal_rabi_frequency
                    rabi_frequency_derivative = 0   
            rabi_frequencies[j] = rabi_frequency
            
            frequency_offset = frequency_axis[i] - (frequency_center - 0.5*frequency_sweep_width + frequency_sweep_width * time_axis[j] / pulse_length)
            frequency_offset_derivative = -frequency_sweep_width / pulse_length
            
            if rabi_frequency == 0 and frequency_offset == 0:
                adiabaticity_value = 0
            else:
                adiabaticity_value = (rabi_frequency**2 + frequency_offset**2)**1.5 / np.abs(rabi_frequency*frequency_offset_derivative - frequency_offset*rabi_frequency_derivative)
            adiabaticity_array[i][j] = adiabaticity_value
            
    adiabaticities = np.amin(adiabaticity_array, axis=1)
    print(np.argmin(adiabaticity_array, axis=1))
    excitation_probabilities = 1 - np.exp(-0.5*np.pi * adiabaticities)
    return microwave_frequencies, rabi_frequencies, adiabaticities, excitation_probabilities, time_axis, frequency_axis 


def plot_results(microwave_frequencies, rabi_frequencies, adiabaticities, excitation_probabilities, time_axis, frequency_axis):
    
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    rcParams['xtick.direction'] = 'out'
    rcParams['ytick.direction'] = 'out'
    rcParams['axes.facecolor']= 'white'
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = 'Arial'
    rcParams['font.size'] = 16
    
    fig = plt.figure(figsize = [9, 9], facecolor='w', edgecolor='w')
    
    axes = fig.add_subplot(2, 2, 1)
    axes.plot(time_axis, rabi_frequencies, 'k-')
    axes.set_xlabel(r'Time (ns)')
    axes.set_ylabel('Rabi frequency (GHz)')
    axes.set_xlim(min(time_axis), max(time_axis))
    
    axes = fig.add_subplot(2, 2, 3)
    axes.plot(time_axis, microwave_frequencies, 'k-')
    axes.set_xlabel(r'Time (ns)')
    axes.set_ylabel('Microwave frequency (GHz)')
    axes.set_xlim(min(time_axis), max(time_axis))
    
    axes = fig.add_subplot(2, 2, 2)
    axes.plot(frequency_axis, adiabaticities, 'k-')
    axes.set_xlabel(r'Frequency (GHz)')
    axes.set_ylabel('Adiabaticity (arb. u.)')
    axes.set_xlim(min(frequency_axis), max(frequency_axis))
    axes.set_ylim(0, max(adiabaticities)+0.5)
    
    axes = fig.add_subplot(2, 2, 4)
    axes.plot(frequency_axis, excitation_probabilities, 'k-')
    axes.set_xlabel(r'Frequency (GHz)')
    axes.set_ylabel('Excitation probability (arb. u.)')
    axes.set_xlim(min(frequency_axis), max(frequency_axis))
    axes.set_ylim(0, 1.1)
    
    fig.tight_layout()
    plt.show() 


# test
if __name__ == '__main__':
    
    # frequency_center = 9.400
    # frequency_sweep_width = 0.630
    # pulse_length = 128
    # rise_time = 40
    # critical_adiabaticity = 5.0
    # #critical_adiabaticity = 2 * np.log(2) / np.pi
    
    frequency_center = 34.427697 
    frequency_sweep_width = 0.100
    pulse_length = 82 
    rise_time = 20
    #critical_adiabaticity = 5
    critical_adiabaticity = 0.1
    
    microwave_frequencies, rabi_frequencies, adiabaticities, excitation_probabilities, time_axis, frequency_axis  = chirp_excitation_bandwidth(frequency_center, frequency_sweep_width, pulse_length, rise_time, critical_adiabaticity)
    
    plot_results(microwave_frequencies, rabi_frequencies, adiabaticities, excitation_probabilities, time_axis, frequency_axis)