import numpy as np
from math import sqrt
from supplement.definitions import const


class Spin:
    ''' Spin ''' 

    def __init__(self, g, n, I, Abund, A, gStrain, AStrain, lwpp, T1, g_anisotropy_in_dipolar_coupling):
        self.g = g
        self.n = n
        self.I = I
        self.Abund = Abund
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
        ''' Computes the number of EPR transitions and their relative intensities '''
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
        ''' Simulates the homogeneous broadering '''
        return const['MHz2GHz'] * const['pp2std'] * self.lwpp * np.random.normal(0,1,size)

    def g_effective(self, field_orientations, size):
        ''' Computes effective g-values for given magnetic field orientations'''
        if self.gStrain.size:
            if (self.gStrain[0] != 0) or (self.gStrain[1] != 0) or (self.gStrain[2] != 0):
                g_xx = self.g[0] + const["fwhm2std"] * self.gStrain[0] * np.random.normal(0,1,size)
                g_yy = self.g[1] + const["fwhm2std"] * self.gStrain[1] * np.random.normal(0,1,size)
                g_zz = self.g[2] + const["fwhm2std"] * self.gStrain[2] * np.random.normal(0,1,size)
                g = np.transpose(np.vstack((g_xx, g_yy, g_zz)))
            else:
                g = np.tile(self.g, (size,1))
        else:
            g = np.tile(self.g, (size,1))
        g_eff_values = np.sqrt(np.sum(g**2 * field_orientations**2, axis=1)) 
        return g_eff_values
   
    def A_effective(self, field_orientations, size, no_nucleus):
        ''' Computes effective A-values for given magnetic field orientations'''
        if self.AStrain.size:
            if (self.AStrain[no_nucleus][0] != 0) or (self.AStrain[no_nucleus][1] != 0) or (self.AStrain[no_nucleus][2] != 0):
                A_xx = const['MHz2GHz'] * (self.A[no_nucleus][0] + const["fwhm2std"] * self.AStrain[no_nucleus][0] * np.random.normal(0,1,size))
                A_yy = const['MHz2GHz'] * (self.A[no_nucleus][1] + const["fwhm2std"] * self.AStrain[no_nucleus][1] * np.random.normal(0,1,size))
                A_zz = const['MHz2GHz'] * (self.A[no_nucleus][2] + const["fwhm2std"] * self.AStrain[no_nucleus][2] * np.random.normal(0,1,size))
                A = np.transpose(np.vstack((A_xx, A_yy, A_zz)))
            else:
                A = np.tile(const['MHz2GHz'] * self.A[no_nucleus], (size,1))
        else:
            A = np.tile(const['MHz2GHz'] * self.A[no_nucleus], (size,1))
        A_eff_values = np.sqrt(np.sum(A**2 * field_orientations**2, axis=1)) 
        return A_eff_values

    def res_freq(self, field_orientations, field_value):
        ''' Computes resonance frequencies for given magnetic field orientations'''
        # Number of field directions
        num_field_orientations = field_orientations.shape[0]
        # Effective g-factors
        g_eff_values = self.g_effective(field_orientations, num_field_orientations)
        g_eff = g_eff_values.reshape(num_field_orientations, 1)
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
            for i in range(self.n.size):
                Aeff_values = self.A_effective(field_orientations, num_field_orientations, i)
                Aeff = Aeff_values.reshape(num_field_orientations, 1)
                I_eq = self.I[i] * float(self.n[i])
                m_eq = np.arange(-1*I_eq, I_eq + 1, step=1, dtype=float)
                num_res_freq_for_single_n = int(2 * self.I[i] * self.n[i] + 1)
                m = np.repeat(m_eq, num_repeat)
                num_repeat *= num_res_freq_for_single_n
                num_tile = int(num_tile / num_res_freq_for_single_n)
                m = np.tile(m, num_tile)
                fh = Aeff * m
                w = np.where(np.random.rand(num_field_orientations) <= self.Abund[i], 1, 0)
                w = w.reshape(num_field_orientations, 1)
                w = np.tile(w, self.num_res_freq)
                fh = fh * w
                f += fh
        return f, g_eff_values

    def quantization_axis(self, field_orientations, g_eff=[]):
        ''' Computes quantization axes for given magnetic field orientations'''
        # Number of field directions
        num_field_orientations = field_orientations.shape[0]
        # Effective g-factors
        if g_eff == []:
            g_eff = self.g_effective(field_orientations, num_field_orientations)
        else:
            g_eff = g_eff.reshape((num_field_orientations,1))
        # Principal components of the g-matrix
        g = np.tile(self.g, (num_field_orientations,1)) 
        quantization_axes = (g / g_eff)**2 * field_orientations
        return quantization_axes
    
    def __eq__(self, other):
        ''' The equality operator '''
        if isinstance(other, self.__class__):
            if (self.g == other.g).all() and \
               (self.n == other.n).all() and \
               (self.I == other.I).all() and \
               (self.Abund == other.Abund).all() and \
               (self.A == other.A).all() and \
               (self.gStrain == other.gStrain).all() and \
               (self.AStrain == other.AStrain).all() and \
               (self.lwpp == other.lwpp) and \
               (self.T1 == other.T1) and \
               (self.g_anisotropy_in_dipolar_coupling == other.g_anisotropy_in_dipolar_coupling):
                return True
        else:
            return False 