''' Spin class '''

import numpy as np
from math import sqrt

from supplement.definitions import const

class Spin:

    def __init__(self, g, n, I, A, gStrain, AStrain, lwpp, T1, g_anisotropy_in_dipolar_coupling):
        self.g = g
        self.n = n
        self.I = I
        self.A = A
        self.gStrain = gStrain
        self.AStrain = AStrain
        self.lwpp = lwpp
        self.T1 = T1
        self.g_anisotropy_in_dipolar_coupling = g_anisotropy_in_dipolar_coupling
        self.num_trans = 0
        self.num_res_freq = 0
        self.int_res_freq = []
        self.count_transitions()
    
    def count_transitions(self):
        ''' Compute the number of EPR transitions and their relative intensities '''
        self.num_trans = 1
        self.num_res_freq = 1
        self.int_res_freq = np.array([1.0])
        if self.n.size:
            for i in range(len(self.n)):
                self.num_trans *= int((2 * self.I[i] + 1) ** self.n[i]) 
                self.num_res_freq *= int(2 * self.I[i] * self.n[i] + 1)
                idx_I = int(2 * self.I[i] - 1)
                idx_n = int(self.n[i] - 1)
                int_res_freq_for_single_n = const['relative_intensities'][idx_I][idx_n]
                result = []
                for j in int_res_freq_for_single_n:
                    result.extend(np.dot(j, self.int_res_freq))
                self.int_res_freq = np.array(result)
        self.int_res_freq /= float(self.num_trans)

    def line_broadening(self, size):
        ''' Simulates homogeneous broadering '''
        return const['MHz2GHz'] * self.lwpp * np.random.normal(0,1,size)

    def g_effective(self, field_orientations, size):
        ''' Computes effective g-values for given magnetic field orientations'''
        if self.gStrain.size:
            if (self.gStrain[0] != 0) or (self.gStrain[1] != 0) or (self.gStrain[2] != 0):
                gxx = self.g[0] + const["fwhm2sd"] * self.gStrain[0] * np.random.normal(0,1,size)  
                gyy = self.g[1] + const["fwhm2sd"] * self.gStrain[1] * np.random.normal(0,1,size)
                gzz = self.g[2] + const["fwhm2sd"] * self.gStrain[2] * np.random.normal(0,1,size)
                g = np.transpose(np.vstack((gxx, gyy, gzz)))  #np.columnstack() does the same
                g_eff = np.sqrt(np.sum(g**2 * field_orientations**2, axis=1))
            else:
                g_eff = np.sqrt(np.sum(self.g**2 * field_orientations**2, axis=1))
        else:
            g_eff = np.sqrt(np.sum(self.g**2 * field_orientations**2, axis=1))
        return g_eff.reshape(size, 1)
    
    def A_effective(self, field_orientations, size, no_nucleus):
        ''' Computes effective A-values for given magnetic field orientations'''
        if self.AStrain.size:
            if (self.AStrain[no_nucleus][0] != 0) or (self.AStrain[no_nucleus][1] != 0) or (self.AStrain[no_nucleus][2] != 0):
                Axx = self.A[no_nucleus][0] + const["fwhm2sd"] * self.A[no_nucleus][0] * np.random.normal(0,1,size)  
                Ayy = self.A[no_nucleus][1] + const["fwhm2sd"] * self.A[no_nucleus][1] * np.random.normal(0,1,size)
                Azz = self.A[no_nucleus][2] + const["fwhm2sd"] * self.A[no_nucleus][2] * np.random.normal(0,1,size)
                A = np.transpose(np.vstack((Axx, Ayy, Azz)))
                A_eff = const['MHz2GHz']* np.sqrt(np.sum(A**2 * field_orientations**2, axis=1))
            else:
                A_eff = const['MHz2GHz']* np.sqrt(np.sum(self.A[no_nucleus]**2 * field_orientations**2, axis=1))
        else:
            A_eff = const['MHz2GHz']* np.sqrt(np.sum(self.A[no_nucleus]**2 * field_orientations**2, axis=1))
        return A_eff.reshape(size, 1)

    def res_freq(self, field_orientations, field_value):
        ''' Computes resonance frequencies for given magnetic field orientations'''
        # Number of field directions
        num_field_orientations = field_orientations.shape[0]
        # Effective g-factors
        g_eff = self.g_effective(field_orientations, num_field_orientations)
        # Resonance frequencies
        f = []
        if self.n.size == 0:
            # Electron Zeeman 
            f = const['Fez'] * field_value * g_eff
            # Inhomogenious broadering
            f += self.line_broadening((num_field_orientations,1))
        else:  
            # Electron Zeeman 
            fz = const['Fez'] * field_value * g_eff
            f = np.tile(fz, self.num_res_freq)
            # Inhomogenious broadering
            fb = self.line_broadening((num_field_orientations, self.num_res_freq))
            f += fb
            # Hyperfine coupling
            num_repeat = 1
            num_tile = self.num_res_freq
            for i in range(len(self.n)):
                Aeff = self.A_effective(field_orientations, num_field_orientations, i)
                I_eq = self.I[i] * self.n[i]
                m_eq = np.arange(-I_eq, I_eq + 1)
                num_res_freq_for_single_n = int(2 * self.I[i] * self.n[i] + 1)
                m = np.repeat(m_eq, num_repeat)
                num_repeat *= num_res_freq_for_single_n
                num_tile /= num_res_freq_for_single_n
                m = np.tile(m, num_tile)
                fh = Aeff * m
                f += fh
        return f, g_eff.flatten()