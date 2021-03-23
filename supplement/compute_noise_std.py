import os
import sys
import numpy as np


def load_signals(filepath):
    t = []
    s = []
    f = []
    file = open(filepath, 'r')
    for line in file:
        str = line.split()
        t.append(float(str[0]))
        s.append(float(str[1]))
        f.append(float(str[2]))
    file.close()
    return np.array(t), np.array(s), np.array(f)
    
def compute_rmsd(s1, s2):
    rmsd = 0.0
    N = s1.size
    for i in range(N):
        rmsd += (s1[i]-s2[i])**2
    rmsd = np.sqrt(rmsd/float(N))
    return rmsd

def chi2(x1, x2, sn=0):
    ''' Calculate chi2 between two signals '''
    chi2 = 0.0
    if sn:
        norm = 1 / sn**2
    else:
        norm = 1.0
    chi2 = norm * np.sum((x1 - x2)**2)  
    return chi2

filepath = 'D:\Project/Software/PeldorFit2021/source_code/examples/nitroxide_biradical_Wband_PELDOR/2021-03-22_13-30/fit_offset XX.dat'
t, s, f = load_signals(filepath)
noise_std = np.std(s-f)   
print(noise_std)
rmsd = compute_rmsd(s,f)
print(rmsd)

noise_std = 0.0016
chi2 = chi2(s, f, noise_std)
print(chi2)
