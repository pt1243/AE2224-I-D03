import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter, welch, hilbert, find_peaks, find_peaks_cwt, peak_widths
from scipy import ndimage
import os
import pandas as pd
from shm_ugw_analysis.data_io.load import load_data, allowed_emitters, allowed_receivers, allowed_frequencies
from shm_ugw_analysis.data_io.signal import Signal, signal_collection
from shm_ugw_analysis.data_io.paths import ROOT_DIR
import pathlib
from numpy.fft import fft, fftfreq, fftshift
from matplotlib.ticker import ScalarFormatter


PLOT_DIR = ROOT_DIR.joinpath('plots')
if not pathlib.Path.exists(PLOT_DIR):
    pathlib.Path.mkdir(PLOT_DIR)

# Note that Welch's method is a cross power spectral density approximation Pxy 
# although it would be applied with respect to itself in the context of a 1D array.

# We may apply scipy.signal.coherence to create a unitless comparison of similarity
# between 2 time series vectors x and y computed using Welch cross power spectral
# density equations: Cxy = abs(Pxy)**2/(Pxx*Pyy)

def maxr(freq, Mag):
    max = 0
    index = 0
    for i in range(0, len(Mag)):
        if freq[i] > (frequency[j]-40)*10**3 and freq[i] < (frequency[j]+40)*10**3:
            if Mag[i] > max:
                max = Mag[i]
                index = i
    return max, index


def fft_and_psd_plots(sc: signal_collection, bin_width, emitter, receiver):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
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
        x_std = (x - np.mean(x))/np.std(x)

        # Standard FFT Method
        ## Option 1: Hanning Window
        '''
        N = len(x)
        window = np.hanning(N)
        x_filtered = x_std * window
        y_fft = np.fft.fft(x_filtered)/N
        x_fft = np.fft.fftfreq(N, d=1/fs)
        y_fft_magnitude = np.abs(y_fft)
        y_fft_magnitude = gaussian_filter1d(y_fft_magnitude, sigma=10)
        #y_fft_magnitude_dB = 10*np.log10(y_fft_magnitude)
        '''
        ## Option 2: 
        N = len(x)
        y_fft = fft(x_std)
        x_fft = fftfreq(N, d=1/fs)
        y_fft_magnitude = np.abs(y_fft)
        y_fft_magnitude = gaussian_filter1d(y_fft_magnitude, sigma=1)
        y_fft_magnitude_dB = 10*np.log10(y_fft_magnitude)
        y_fft_magnitude_dB = gaussian_filter1d(y_fft_magnitude_dB, sigma=1)
        # Welch PSD Method
        x_psd_welch, y_psd_welch = welch(x, fs, nperseg=nperseg)
        y_psd_welch_magnitude = np.abs(y_psd_welch)
        y_psd_welch_magnitude_dB = 10*np.log10(y_psd_welch_magnitude)
        '''
        x = electrocardiogram()[2000:4000]
        peaks, _ = find_peaks(x, height=0)
        plt.plot(x)
        plt.plot(peaks, x[peaks], "x")
        plt.plot(np.zeros_like(x), "--", color="gray")
        plt.show()
        '''

        '''
        # Hilber Transform/Envelope
        analytic_signal = hilbert(y_psd_welch_magnitude)
        amplitude_envelope = np.abs(analytic_signal)
        '''
        # Plotting
        ax1.plot(x_fft, y_fft_magnitude_dB, label = f'cycle {s.cycle}')
        ax2.plot(x_psd_welch, y_psd_welch_magnitude_dB, label = f'cycle {s.cycle}')
        #ax2.plot(x_psd_welch, amplitude_envelope, label='envelope')
        
        # Peak Finding
        border_min = -97
        border_max = -50
        psd_welch_peaks_location, _ = find_peaks(y_psd_welch_magnitude_dB, height=(border_min, border_max))
        x_psd_welch_peaks = x_psd_welch[psd_welch_peaks_location]
        y_psd_welch_peaks = y_psd_welch_magnitude_dB[psd_welch_peaks_location]
        print(f'Plot "PSD emitter {emitter} receiver {receiver} frequency {s.frequency} kHz" has peaks of magnitude {y_psd_welch_peaks} at locations {psd_welch_peaks_location}')

        ax2.plot(x_psd_welch_peaks, y_psd_welch_peaks, "x")
        ax2.plot(border_min, "--", color="gray")
        ax2.plot(border_max, ":", color="gray")

        #plt.plot(peaks, x[peaks], "x")
        #results_half = peak_widths(x, peaks, rel_height=0.5)

        # Peak Width Finding
        results_half = peak_widths(y_psd_welch_peaks, x_psd_welch, rel_height=0.5)
        results_full = peak_widths(y_psd_welch_peaks, x_psd_welch, rel_height=1)
        print(results_half[0])
        print(results_full[0])
        ax2.hlines(*results_half[1:], color="C2")
        ax2.hlines(*results_full[1:], color="C3")
        
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
    plt.savefig(file_path, dpi = 500)
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
                fft_and_psd_plots(sc, 5000, emitter, receiver)
            except Exception:
                continue

# for s in sc:
#     fft_and_psd_plots(s, 5000)

# fft_and_psd_plots(Signal('1000', 'received', 1, 4, 100), 5000)