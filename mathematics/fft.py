import math
import numpy as np
from scipy import signal


def dc_offset_correction(sig, dc_offset=0):
    return sig - dc_offset * np.ones(sig.size)


def appodization(sig, appodization_type='none'):
    sig_app = []
    if appodization_type == 'hamming':
        hamming = np.hamming(2*sig.size-1)
        window = hamming[-sig.size:]
        sig_app = window * sig
    else:
        window = np.ones(sig.size)
        sig_app = sig
    return sig_app, window


def zero_padding(t, sig, zero_padding_factor=0):
    t_zp = []
    sig_zp = []
    if (zero_padding_factor <= 1):
        t_zp = t
        sig_zp = sig
    else:
        t_max = t[0] + (t[1]-t[0]) * float((zero_padding_factor) * t.size - 1)
        t_zp = np.linspace(t[0], t_max, zero_padding_factor * t.size)
        sig_zp = np.append(sig, np.zeros((zero_padding_factor-1) * sig.size))
    return t_zp, sig_zp


def scale_first_point(sig, scale_factor=1):
    sig_scaled = sig
    sig_scaled[0] = scale_factor * sig[0]
    return sig_scaled


def fft(t, sig, dc_offset=0, appodization_type='hamming', zero_padding_factor=2, scale_factor_first_point=0.5):
    ''' FFT of a background-free part of a PDS time trace '''
    # Format data
    if isinstance(t, np.ndarray):
        t_in = t
    else:
        t_in = np.array(t)
    if isinstance(sig, np.ndarray):
        sig_in = sig
    else:
        sig_in = np.array(sig)
    
    # Substract the DC offset
    sig_shifted = dc_offset_correction(sig_in, dc_offset)
    
    # Apply the appodization
    sig_app, window = appodization(sig_shifted, appodization_type)
    
    # Apply the zero-padding
    t_zp, sig_zp = zero_padding(t_in, sig_app, zero_padding_factor)

    # Scale first point
    sig_scaled = scale_first_point(sig_zp, scale_factor_first_point)
    
    # Apply FFT
    spc = np.fft.fft(sig_scaled)
    dt = t[1] - t[0]
    f = np.fft.fftfreq(spc.size, dt)
    f_sorted = np.fft.fftshift(f)
    spc_sorted = np.fft.fftshift(spc)
    spc_sorted = np.real(spc_sorted)
    spc_sorted = spc_sorted / np.amax(spc_sorted)

    return f_sorted, spc_sorted