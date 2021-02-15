import time
import sys
import numpy as np
from math import ceil, radians, sin, cos, pi, sqrt

from numpy.lib.histograms import histogram

from mathematics import RotationMatrix


class Simulator():

    def __init__(self, number_of_averages):
        self.nfields = number_of_averages
        self.treshold = 10 ** (-3)
        self.fieldDirA = self.initialize_field()
        self.gValuesA = None
        self.detProbsA = None
        self.pumpProbsA = None
        
        
    
    def initialize_field(self):
        """
        Calculate random orientations of the magnetic field vector
        """
        # n random numbers between 0 and 1, uniformly distributed
        random = np.random.uniform(0, 1, self.nfields)
        # calculate spherical angles phi and xi
        fphi = 2 * pi * random
        fxi_temp1 = np.arccos(random)
        fxi_temp2 = pi-fxi_temp1
        fxi = np.where(random<0.5, fxi_temp1, fxi_temp2)
        # conversion from spherical to cartesian
        x = np.sin(fxi) * np.cos(fphi)
        y = np.sin(fxi) * np.sin(fphi)
        z = np.cos(fxi)
        # create the array [(x[0], y[0], z[0]), (x[1], y[1], z[1]), .....]
        fieldDirA = np.column_stack((x, y, z))
        return fieldDirA

    #TODO: unfinished
    # implementation later, depends on other things
    def get_parameter_values(self):
        pass

    def get_value_from_distribution(self, mean, width, type_of_distribution):
        value = 0
        if(width == 0):
            value = mean
        else:
            if(type_of_distribution == "uniform"):
                value = np.random.uniform(
                    low=mean - 0.5 * width, high=mean + 0.5 * width)
            elif(type_of_distribution == "normal"):
                value = np.random.normal(mean, width)
        return value

    # TODO: unfinished
    def spectrum(self, experiment, spin_system):
        # Determine the maximal and minimal frequencies
        # takes around 5 seconds
        fminmax = self.minmax_resonance_freq(experiment, spin_system)
        fmin = fminmax[0]
        fmax = fminmax[1]
        # Increase slightly the frequency range
        fmin -= 0.1
        fmax += 0.1
        # Set the frequency increment to 1 MHz
        df = 0.001
        # Round up the max and min frequencies to MHz
        fmin = ceil(fmin/df)*df
        fmax = ceil(fmax/df)*df
        # Calculate the number of increments
        Nf = int((fmax - fmin) / df)
        # Correlate frequencies with the indices
        idx0 = int((fmin / df) + 1)
        # The frequency values 
        fValues = fmin + 0.5 * df * (2 * np.arange(Nf)+1) #still don't really understand this
        # Calculate the probabilities
        pValues = np.zeros(Nf)
        print("Nf is: " + str(Nf))
        for i in range(len(spin_system)):
            ## TODO: try to replace for loop with numpy array
            #for f in range(Nf):
            #    # Compute the resonance frequences for the single orientation of the magnetic field
            #    resonance_freq = spin_system[i].resonance_freq(self.fieldDirA[f], experiment)
            #    for k in range(spin_system[i].Ncomp):
            #        idx = int(resonance_freq[k] / df) - idx0
            #        pValues[idx] += spin_system[i].Icomp[k] / spin_system[i].Nstates
            resonance_freq = spin_system[i].resonance_freq_array(self.fieldDirA, self.nfields, experiment)
            idx = (resonance_freq.T / df).astype(int) - idx0
            for k in range(spin_system[i].Ncomp):
                idx_histogram = np.bincount(idx[k], minlength=Nf)
                try:
                    pValues += idx_histogram * spin_system[i].Icomp[k] / spin_system[i].Nstates
                except:
                    print("k: "+ str(k))
                    print("shape of idx_histogram: " +str( np.shape(idx_histogram)))
                    print("shape of pValues: " +str( np.shape(pValues)))
        # Normalize the probabilities
        pMax = np.max(pValues)
        pValues /= pMax
        # Create the spectrum
        spectrum = np.vstack((fValues, pValues))
        return spectrum

        

    #TODO: unfinished
    def peldor_signal(self, optimized_parameter_values, experiment, spin_system, optimization_parameters):

        # Initialize the Rotation Matrix
        rotationMatrix = RotationMatrix()

        # implement Model 1 first
        if(len(spin_system) == 2):
            # implement later, would be convenient if values was a dictionary
            # can this be done outside of Monte Carlo for loop? In old Peldorfit this was called in every iteration
            values = self.get_parameter_values()
            for n in range(self.nfields):
                #get values
                dist = self.get_value_from_distribution(
                    values["mean_dist"], values["width_dist"])
                xi = self.get_value_from_distribution(
                    values["mean_xi"], values["width_xi"])
                phi = self.get_value_from_distribution(
                    values["mean_phi"], values["width_phi"])
                alpha = self.get_value_from_distribution(
                    values["mean_alpha"], values["width_alpha"])
                beta = self.get_value_from_distribution(
                    values["mean_beta"], values["width_beta"])
                gamma = self.get_value_from_distribution(
                    values["mean_gamma"], values["width_gamma"])
                J = self.get_value_from_distribution(
                    values["mean_J"], values["width_J"])
                
                # Read out the g value of spin A
                gValueA = self.gValuesA[n]
                # TODO: Read out the probability of spin A to be excited by the detection pulses
                # TODO: Read out the probability of spin A to be excited by the pump pulse

                # Rotation matrix between the spin A and spin B frames
                rotationMatrix.reset_angles(alpha, beta, gamma)
                # Calculate the direction of the magnetic field in the spin B frame
                rotationMatrix.multiplication_with_vector(self.fieldDirA[n])
                # Calculate the effective g-factor of  spin B
                gValueB = spin_system[1].effective_g(self.fieldDirA[n])
                # Calculate the resonance frequencies of spin B
                resfreqB = spin_system[1].resonance_freq(self.fieldDirA[n], experiment)
                # TODO: Calculate the probability of spin B to be excited by the detection pulses
                # TODO: Calculate the probability of spin B to be excited by the pump pulse





    def directionfromAngles(self, xi, phi):
        # conversion from spherical to cartesian
        direction = np.zeros(3)
        direction[0] = cos(phi) * sin(xi)
        direction[1] = sin(phi) * sin(xi)
        direction[2] = cos(xi)
        return direction
    
    #the root mean square deviation between the simulated and the experimental signal
    def rmsd(x, y): 
        nPoints = np.size(x)
        rmsd = sqrt((np.sum(np.square(x-y)))/nPoints)
    
    def minmax_resonance_freq(self, experiment, spin_system): 
        for i in range(len(spin_system)):
            #resonance_freq = np.zeros((self.nfields, spin_system[i].Ncomp))
            #for f in range(self.nfields):
            #    resonance_freq[f] = spin_system[i].resonance_freq(self.fieldDirA[f], experiment)
            resonance_freq = spin_system[i].resonance_freq_array(self.fieldDirA, self.nfields, experiment)
            min = np.min(resonance_freq)
            max = np.max(resonance_freq)
            if(i == 0):
                fmin = min
                fmax = max
            else:
                if(fmin >min):
                    fmin = min
                if(fmax<max):
                    fmax = max
        return (fmin, fmax)
        
    # TODO: rewrite with numpy
    def excitation_probabilities_spin_A(self, experiments, spinA):
        # TODO: clarify if gValuesA is implemented correctly in C++ version
        for i in range(len(experiments)):
            for f in range(self.nfields):
                # Compute an effective g-factor of the spin A
                gValue = spinA.effective_g(self.fieldDirA[f])
                # Compute resonance frequencies of the spin A
                resfreqA = spinA.resonance_freq(self.fieldDirA[f], experiments[i])
                # Compute the probability of spin A to be excited by the detection pulses
                detProb = self.excitation_probability_detection(resfreqA, experiments[i], spinA)
                # Compute the probability of spin A to be excited by a pump pulse
                pumpProb = self.excitation_probability_pump(resfreqA, experiments[i], spinA)


    
    # Problem: weiß nicht, woher die Formeln kommen
    #klären, wie die Reihenfolge in pulse_lengths is
    def excitation_probability_detection(self, resonance_freq, experiment, spin):
        detProb = 0
        detPiHalfBW = 0.25 / experiment.pulse_lengths[0]
        detPiBW = 0.5 / experiment.pulse_lengths[1]
        for k in range(spin.Ncomp):
            if(detPiHalfBW == detPiBW):
                freqEff = sqrt((experiment.detection_frequency - resonance_freq[k])**2) + detPiHalfBW**2
                prob = ((detPiHalfBW / freqEff)*sin(2*pi * freqEff * experiment.detPiHalfLength))**5
            else:
                freqEff = sqrt((experiment.detection_frequency - resonance_freq[k])**2) + detPiHalfBW**2
                freqEff2 = sqrt((experiment.detection_frequency - resonance_freq[k])**2) + detPiBW**2
                prob = (sqrt((detPiHalfBW / freqEff)* sin(2*pi * freqEff * experiment.pulse_lengths[0])) *
                 ((detPiBW/freqEff2) * sin(0.5 * 2*pi * freqEff2 * experiment.pulse_lengths[1]))**4)
            detProb += abs(spin.Icomp[k] * prob)
        detProb /= spin.Nstates
        return detProb   
   
    # TODO: unfinished
    def excitation_probability_pump(self, resonance_freq, experiment, spin):
        pumpProb = 0
        pumpPiBW = 0.5 / experiment.pulse_lengths[3]
        for k in range(spin.Ncomp):
            freqEff = sqrt((experiment.pump_frequency-resonance_freq[k])**2) + pumpPiBW**2
            prob = (pumpPiBW / freqEff) * sin(0.5 * 2*pi * freqEff * experiment.pulse_lengths[3])**2
            pumpProb += abs(spin.Icomp[k] * prob)
        pumpProb /= spin.Nstates
        return pumpProb