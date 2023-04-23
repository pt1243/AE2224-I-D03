import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter, welch, hilbert, find_peaks, find_peaks_cwt, peak_widths, csd, coherence
from scipy import ndimage
import os
import pandas as pd
from shm_ugw_analysis.data_io.load import load_data, allowed_emitters, allowed_receivers, allowed_frequencies
from shm_ugw_analysis.data_io.signal import Signal, signal_collection, InvalidSignalError
from shm_ugw_analysis.data_io.paths import ROOT_DIR
import pathlib
from numpy.fft import fft, fftfreq, fftshift
from matplotlib.ticker import ScalarFormatter

# TROUBLESHOOTING TO-DO LOG:
# (1) fft_and_psd_plots function should be turned into a psd_plots only function given that's how we produce our indices
# (2) verifying normalization equation for x
# (3) key issue with ax1.___ and ax2.___ implementation, subclasses not loading (check matplotlib implementation in virtual environment)
# (4) plt.close() implementation incorrect? obtaining error:
# RuntimeWarning: More than 30 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) 
# are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`). 
# Consider using `matplotlib.pyplot.close()`. 
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10)) 
# Note: default value for RuntimeWarning is for more than 20 figures opened
# (5) need to treat data to find average of primary and secondary peaks across all frequency bands 
# (6) need to adjust cross spectral density and coherence mapping so we cycle through frequencies whilst retaining a baseline frequency
# (7) all paths (e=3 -> r=4) are not sufficiently smooth at primary peak leading to several maxima values being identified

PLOT_DIR = ROOT_DIR.joinpath('plots')
if not pathlib.Path.exists(PLOT_DIR):
    pathlib.Path.mkdir(PLOT_DIR)

# Note that Welch's method is a cross power spectral density approximation Pxy 
# although it would be applied with respect to itself in the context of a 1D array.

# We may apply scipy.signal.coherence to create a unitless comparison of similarity
# between 2 time series vectors x and y computed using Welch cross power spectral
# density equations: Cxy = abs(Pxy)**2/(Pxx*Pyy)

# IDEALLY SHOULD ONLY BE PSD PLOTTING FUNCTION WITH MAX/MIN & SMOOTHING

def find_peak(x_psd_welch, y_psd_welch_magnitude_dB, border_min = -97, border_max = -50):
    ## NEED TO ENSURE BORDERS ARE WELL-DEFINED FOR EFFECTIVE PEAK SEARCHING
    psd_welch_peaks_location, _ = find_peaks(y_psd_welch_magnitude_dB, height=(border_min, border_max))
    x_psd_welch_peaks = x_psd_welch[psd_welch_peaks_location]
    y_psd_welch_peaks = y_psd_welch_magnitude_dB[psd_welch_peaks_location]
    print(f'Plot "PSD emitter {emitter} receiver {receiver} frequency {frequency} kHz" has peaks of magnitude {y_psd_welch_peaks} at locations {psd_welch_peaks_location}')
    #x_psd_welch_peaks = np.ndarray.flatten(x_psd_welch_peaks)
    print(x_psd_welch_peaks)
    #y_psd_welch_peaks = np.ndarray.flatten(y_psd_welch_peaks)
    print(y_psd_welch_peaks)
    local_peaks = np.concatenate([x_psd_welch_peaks, y_psd_welch_peaks])
    print(f'Local Peak Matrix: {local_peaks}')

    # Peak Width Finding
    #results_half = peak_widths(y_psd_welch_peaks, x_psd_welch, rel_height=0.5)
    #results_full = peak_widths(y_psd_welch_peaks, x_psd_welch, rel_height=1)
    #print(results_half[0])
    #print(results_full[0])
    return x_psd_welch_peaks, y_psd_welch_peaks, local_peaks

def psd_plot_peak_finding(sc: signal_collection, bin_width, emitter, receiver):
    matrix_peaks = np.empty([7, 18])
    for s in sc:
        # Signal Segmentation
        s: Signal
        x = s.x
        t = s.t
        # Parameters
        fs = s.sample_frequency
        #print(f'fs={fs}')
        nperseg = int(fs/bin_width)
        # Normalizing
        ## IS THIS A CORRECT NORMALIZATION METHOD?
        x_std = (x - np.mean(x))/np.std(x)

        # Welch PSD Method
        x_psd_welch, y_psd_welch = welch(x, fs, nperseg=nperseg)
        y_psd_welch_magnitude = np.abs(y_psd_welch)
        y_psd_welch_magnitude_dB = 10*np.log10(y_psd_welch_magnitude)
        
        # Peak Finding
        ## NEED TO ENSURE BORDERS ARE WELL-DEFINED FOR EFFECTIVE PEAK SEARCHING
        border_min = -97
        border_max = -50
        x_psd_welch_peaks, y_psd_welch_peaks, local_peaks = find_peak(x_psd_welch, y_psd_welch_magnitude_dB, border_min, border_max)
        np.append(matrix_peaks, local_peaks, axis=0)
        print(f'MATRIX PEAKS: {matrix_peaks}')
    return 


def fft_and_psd_plots(sc: signal_collection, bin_width, emitter, receiver):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    plt.rcParams.update({'figure.max_open_warning': 10})
    matrix_peaks = np.empty([7, 18])
    for s in sc:
        # Signal Segmentation
        s: Signal
        x = s.x
        t = s.t
        # Parameters
        fs = s.sample_frequency
        #print(f'fs={fs}')
        nperseg = int(fs/bin_width)
        # Normalizing
        ## IS THIS A CORRECT NORMALIZATION METHOD?
        x_std = (x - np.mean(x))/np.std(x)

        # Welch PSD Method
        x_psd_welch, y_psd_welch = welch(x, fs, nperseg=nperseg)
        y_psd_welch_magnitude = np.abs(y_psd_welch)
        y_psd_welch_magnitude_dB = 10*np.log10(y_psd_welch_magnitude)
        
        # Plotting
        #ax1.plot(x_fft, y_fft_magnitude_dB, label = f'cycle {s.cycle}')
        #ax2.plot(x_psd_welch, y_psd_welch_magnitude_dB, label = f'cycle {s.cycle}')
        #ax2.plot(x_psd_welch, amplitude_envelope, label='envelope')
        
        # Peak Finding
        ## NEED TO ENSURE BORDERS ARE WELL-DEFINED FOR EFFECTIVE PEAK SEARCHING
        border_min = -97
        border_max = -50
        x_psd_welch_peaks, y_psd_welch_peaks = find_peak(x_psd_welch, y_psd_welch_magnitude_dB, border_min, border_max)
        #np.append(matrix_peaks, local_peaks, axis=0)
        print(f'MATRIX PEAKS: {matrix_peaks}')
        
        ax2.plot(x_psd_welch_peaks, y_psd_welch_peaks, "x")
        ax2.plot(border_min, "--", color="gray")
        ax2.plot(border_max, ":", color="gray")

        #plt.plot(peaks, x[peaks], "x")
        #results_half = peak_widths(x, peaks, rel_height=0.5)

        # Peak Width Plotting
        #ax2.hlines(*results_half[1:], color="C2")
        #ax2.hlines(*results_full[1:], color="C3")

        fig.show()

        # Cross PSD
        # f, Pxy = csd(x_freq_baseline, x_freq_all_but_baseline)
        # f, Cxy = coherence(x_freq_baseline, x_freq_all_but_baseline)
        ######## used to be <-- indented #######
    ax1.grid()  
    ax1.set_xlabel('Frequency [Hz]')
    ax1.set_ylabel('Amplitude')
    ax1.set_xlim(0.5*10**5, 7*10**5)
    ax1.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax1.ticklabel_format(style="sci", axis="x", scilimits=(0,0))
    ax1.legend()
    ax1.set_title(f'FFT emitter {emitter} receiver {receiver} frequency {s.frequency} kHz')    
    ax2.grid()
    ax2.set_xlabel('Frequency [Hz]')
    ax2.set_ylabel('Power (W)')
    ax2.set_xlim(0.5*10**5, 7*10**5)
    ax2.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax2.ticklabel_format(style="sci", axis="x", scilimits=(0,0))
    ax2.legend()
    ax2.set_title(f'PSD emitter {emitter} receiver {receiver} frequency {s.frequency} kHz')
    file_path = os.path.join(PLOT_DIR, f'FFT+PSD_emitter_{emitter}_receiver_{receiver}_frequency_{s.frequency}_kHz.png')
    fig.savefig(file_path, dpi=500)
    # plt.savefig(file_path, dpi = 500)
    fig.clear()
    return

#cycles=['0', '1', '10000', '20000', '30000', '40000', '50000', '60000', '70000']
cycles = ['20000']
signal_types=['excitation']
# emitters=list(allowed_emitters)
emitters = [1]
# receivers=list(allowed_receivers)
receivers = [4]
frequencies=list(allowed_frequencies)
# frequencies = [100]

for frequency in frequencies:
    for emitter in emitters:
        for receiver in receivers:
            try:
                sc = signal_collection(cycles=tuple(cycles), signal_types=tuple(signal_types), emitters=[emitter], receivers=[receiver], frequencies=[frequency])
                psd_plot_peak_finding(sc, 10000, emitter, receiver)
            except InvalidSignalError as e:
                continue

# for s in sc:
#     fft_and_psd_plots(s, 5000)

# fft_and_psd_plots(Signal('1000', 'received', 1, 4, 100), 5000)