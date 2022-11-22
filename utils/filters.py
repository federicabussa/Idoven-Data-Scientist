import numpy as np
from scipy import signal as sig


def filter_bandpass(
        y, 
        f_hp = 0.5, 
        f_lp = 45,
        sample_rate = 500, 
        order = 5
    ):

    if y.shape[0] < y.shape[1]:
        y = y.T
    ncols = y.shape[1]
    nyq = 0.5 * sample_rate
    b_h, a_h = sig.butter(order, f_hp / nyq, btype='highpass')
    b_l, a_l = sig.butter(order, f_lp / nyq, btype='lowpass')
    y_filt = np.zeros(y.shape, dtype=y.dtype) 
    for c in range(0, ncols):
        y_tmp = sig.filtfilt(b_h, a_h, y[:, c])
        y_filt[:, c] = sig.filtfilt(b_l, a_l, y_tmp)
    
    return y_filt


def filter_notch(
            y, 
            f_notch = 50.0,
            sample_rate = 500, 
            order = 4
    ):
    
    cutoff_notch =  np.array([f_notch - 0.2, f_notch + 0.2], dtype=np.float64) / sample_rate * 2.0
    b_n, a_n = sig.butter(order, cutoff_notch, 'bandstop')
    y_filt = sig.filtfilt(b_n, a_n, y)
    
    return np.array(y_filt, dtype=y.dtype)
