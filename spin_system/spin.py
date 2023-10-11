import numpy as np
from supplement.definitions import const


class Spin:
    """Spin."""

    def __init__(self):
        self.parameter_names = {
             "g": "float",
             "gStrain": "float",
             "n": "int",
             "I": "float",
             "Abund": "float",
             "A": "float",
             "AStrain": "float",
             "lwpp": "float",
             "T1": "float",
             "g_anisotropy_in_dipolar_coupling": "bool"
            }
    
    
    def set_parameters(self, parameter_values):
        """Set the EPR parameters."""
        self.g = np.array(parameter_values["g"])
        self.gStrain = np.array(parameter_values["gStrain"])
        self.n = np.array(parameter_values["n"])
        self.I = np.array(parameter_values["I"])
        self.Abund = np.array(parameter_values["Abund"])
        self.A = np.array(parameter_values["A"])
        self.AStrain = np.array(parameter_values["AStrain"])
        self.lwpp = parameter_values["lwpp"]
        self.T1 = parameter_values["T1"]
        self.g_anisotropy_in_dipolar_coupling = parameter_values["g_anisotropy_in_dipolar_coupling"]
        # Perform several dimensionality checks
        if self.g.size != 3:
            raise ValueError("Invalid number of elements in g!")
            sys.exit(1)
        if self.gStrain.size != 0 and self.gStrain.size != 3:
            raise ValueError("Invalid number of elements in gStrain!")
            sys.exit(1)
        if self.I.size != self.n.size:
            raise ValueError("Number of elements in n and I must be equal!")
            sys.exit(1)
        if self.A.size != 3 * self.n.size:
            raise ValueError("Invalid number of elements in A!")
            sys.exit(1)
        if self.AStrain.size != 0 and self.AStrain.size != self.A.size: 
            raise ValueError("Invalid number of elements in AStrain!")
            sys.exit(1)
        # Count the number of EPR transitions for a single orientation of the magnetic field
        self.count_transitions()
    
    
    def count_transitions(self):
        """Compute the number of EPR transitions and their relative intensities."""
        self.num_trans = 1
        self.num_res_freq = 1
        self.int_res_freq = np.array([1.0])
        if self.n.size:
            for i in range(len(self.n)):
                self.num_trans *= int((2 * self.I[i] + 1) ** self.n[i]) 
                self.num_res_freq *= int(2 * self.I[i] * self.n[i] + 1)
                idx_I = int(2 * self.I[i] - 1)
                idx_n = int(self.n[i] - 1)
                int_res_freq_for_single_n = const["relative_intensities"][idx_I][idx_n]
                result = []
                for j in int_res_freq_for_single_n:
                    result.extend(np.dot(j, self.int_res_freq))
                self.int_res_freq = np.array(result)
        self.int_res_freq /= float(self.num_trans)
    
    
    def res_freq(self, field_orientations, field_value):
        """Compute resonance frequencies for given magnetic field orientations."""
        # Effective g-factors
        g_eff = self.g_effective(field_orientations)
        g_eff = np.expand_dims(g_eff, -1)
        # Resonance frequencies
        if self.n.size == 0:
            # Electron Zeeman 
            f_zeeman = const["Fez"] * field_value * g_eff
            f_res = f_zeeman
            # Inhomogenious broadering
            f_broad = self.line_broadening((field_orientations.shape[0], 1))
            f_res += f_broad
        else:  
            # Electron Zeeman 
            f_zeeman = const["Fez"] * field_value * g_eff
            f_res = np.tile(f_zeeman, (1, self.num_res_freq))
            # Inhomogenious broadering
            f_broad = self.line_broadening((field_orientations.shape[0], self.num_res_freq))
            f_res += f_broad
            # Hyperfine coupling
            num_repeat, num_tile = 1, self.num_res_freq
            for i in range(self.n.size):
                A_eff = self.A_effective(field_orientations, i)
                A_eff = np.expand_dims(A_eff, -1)
                I_eq = self.I[i] * self.n[i]
                m_eq = np.arange(-I_eq, I_eq + 1, step=1, dtype=float)
                num_freq_single_n = 2 * self.I[i] * self.n[i] + 1
                m = np.repeat(m_eq, num_repeat)
                num_repeat *= num_freq_single_n
                num_tile = int(num_tile / num_freq_single_n)
                m = np.tile(m, num_tile)
                f_hfi = A_eff * m
                w = np.where(np.random.rand(field_orientations.shape[0]) <= self.Abund[i], 1.0, 0.0)
                w = np.expand_dims(w, -1)
                w = np.tile(w, self.num_res_freq)
                f_hfi = f_hfi * w
                f_res += f_hfi
        return f_res, np.squeeze(g_eff, axis=1)
    
    
    def line_broadening(self, size):
        """Simulate homogeneous EPR line broadering."""
        return const["MHz2GHz"] * const["pp2std"] * self.lwpp * np.random.normal(0, 1, size = size)
    
    
    def g_effective(self, field_orientations):
        """Compute effective g-values for given magnetic field orientations."""
        num_field_orientations = field_orientations.shape[0]
        if self.gStrain.size != 0:
            if (self.gStrain[0] != 0) or (self.gStrain[1] != 0) or (self.gStrain[2] != 0):
                g = self.g + const["fwhm2std"] * self.gStrain * \
                    np.random.normal(0, 1, size=(num_field_orientations, 3))
            else:
                g = np.tile(self.g, (num_field_orientations, 1))
        else:
            g = np.tile(self.g, (num_field_orientations, 1))   
        g_eff = np.sqrt(np.sum(g**2 * field_orientations**2, axis=1)) 
        return g_eff
    
    
    def A_effective(self, field_orientations, idx_nucleus):
        """Compute effective A-values for given magnetic field orientations."""
        num_field_orientations = field_orientations.shape[0]
        if self.AStrain.size != 0:
            if (self.AStrain[idx_nucleus][0] != 0) or \
                (self.AStrain[idx_nucleus][1] != 0) or \
                (self.AStrain[idx_nucleus][2] != 0):
                A = const["MHz2GHz"] * (self.A[idx_nucleus] + \
                    const["fwhm2std"] * self.AStrain[idx_nucleus] * \
                    np.random.normal(0, 1, size=(num_field_orientations, 3)))
            else:
                A = np.tile(const["MHz2GHz"] * self.A[idx_nucleus], (num_field_orientations, 1))
        else:
            A = np.tile(const["MHz2GHz"] * self.A[idx_nucleus], (num_field_orientations, 1))
        A_eff = np.sqrt(np.sum(A**2 * field_orientations**2, axis=1)) 
        return A_eff
    
    
    def quantization_axis(self, field_orientations, g_eff = []):
        """Compute quantization axes for given magnetic field orientations."""
        num_field_orientations = field_orientations.shape[0]
        if g_eff == []:
            g_eff = self.g_effective(field_orientations)
        g_eff = np.expand_dims(g_eff, -1)
        g_principal = np.tile(self.g, (num_field_orientations, 1)) 
        quantization_axes = (g_principal / g_eff)**2 * field_orientations
        return quantization_axes
    
    
    def __eq__(self, other):
        """Equality operator."""
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