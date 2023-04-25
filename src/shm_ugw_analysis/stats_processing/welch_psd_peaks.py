import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from scipy.signal import welch, find_peaks
from ..data_io.load_signals import (
    load_data,
    allowed_emitters,
    allowed_receivers, 
    allowed_frequencies, 
    allowed_cycles,
    Signal, 
    signal_collection,
    frequency_collection,
    path_collection, 
    InvalidSignalError
)
from ..data_io.paths import ROOT_DIR
import pathlib
from matplotlib.ticker import ScalarFormatter
from typing import Optional, Iterable, Sequence

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
    print('here')
    ## NEED TO ENSURE BORDERS ARE WELL-DEFINED FOR EFFECTIVE PEAK SEARCHING
    psd_welch_peaks_location, _ = find_peaks(y_psd_welch_magnitude_dB, height=(border_min, border_max))
    x_psd_welch_peaks = x_psd_welch[psd_welch_peaks_location]
    y_psd_welch_peaks = y_psd_welch_magnitude_dB[psd_welch_peaks_location]
    print(f'Peaks of magnitude {y_psd_welch_peaks} at locations {psd_welch_peaks_location}')
    x_psd_welch_peaks = np.ndarray.flatten(x_psd_welch_peaks)
    print(x_psd_welch_peaks)
    y_psd_welch_peaks = np.ndarray.flatten(y_psd_welch_peaks)
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
    #matrix_peaks = np.empty([7, 18])
    print('here outside loop')
    for s in sc:
        print(f'here inside loop, {s = }')
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
        #x_psd_welch_peaks, y_psd_welch_peaks, local_peaks = find_peak(x_psd_welch, y_psd_welch_magnitude_dB, border_min, border_max)
        find_peak(x_psd_welch, y_psd_welch_magnitude_dB, border_min, border_max)
        #np.append(matrix_peaks, local_peaks, axis=0)
        #print(f'MATRIX PEAKS: {matrix_peaks}')
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
    file_path = PLOT_DIR.joinpath(f'FFT+PSD_emitter_{emitter}_receiver_{receiver}_frequency_{s.frequency}_kHz.png')
    fig.savefig(file_path, dpi=500)
    # plt.savefig(file_path, dpi = 500)
    fig.clear()

def psd_welch(s: Signal, bin_width: int | float):
    fs = s.sample_frequency
    nperseg = int(fs/bin_width)

    x_psd_welch: np.ndarray
    y_psd_welch: np.ndarray

    x_psd_welch, y_psd_welch = welch(s.x, fs, nperseg=nperseg)
    return np.array((x_psd_welch, y_psd_welch))


def find_psd_peaks(
        s: Signal,
        psd: np.ndarray,
        lower_bound: int = -97,
        upper_bound: int = -50,
        distance: Optional[int] = None,
) -> np.ndarray:
    """Calculate the peak locations of the Welch PSD for one specific signal.
    
    Returns a (2, N) ndarray, with N being the number of peaks found, the first subarray being the peak frequencies,
    and the second subarray being the peak magnitudes in dB.
    """

    x_psd_welch: np.ndarray
    y_psd_welch: np.ndarray
    psd_welch_peaks_location: np.ndarray

    x_psd_welch, y_psd_welch = psd[0], psd[1]
    y_psd_welch_magnitude: np.ndarray = np.abs(y_psd_welch)
    y_psd_welch_magnitude_dB: np.ndarray = 10*np.log10(y_psd_welch_magnitude)

    psd_welch_peaks_location, _ = find_peaks(y_psd_welch_magnitude_dB, height=(lower_bound, upper_bound), distance=distance)
    x_psd_welch_peaks: np.ndarray = x_psd_welch[psd_welch_peaks_location].flatten()
    y_psd_welch_peaks: np.ndarray = y_psd_welch_magnitude_dB[psd_welch_peaks_location].flatten()
    peaks_array = np.array((psd_welch_peaks_location, y_psd_welch_peaks))
    print(f'{s} has peaks of magnitude {y_psd_welch_peaks} at locations {psd_welch_peaks_location}')

    return peaks_array


def create_and_save_fig(s: Signal, peaks_array: np.ndarray, psd: np.ndarray) -> None:
    """Plot and save the Welch PSD peaks for one signal."""
    PLOT_DIR = ROOT_DIR.joinpath('plots')
    if not pathlib.Path.exists(PLOT_DIR):
        pathlib.Path.mkdir(PLOT_DIR)
    print(s)
    fig, ax = plt.subplots(1, 1, figsize=(18, 10))
    ax.plot(psd[0], 10*np.log10(psd[1]))
    ax.plot(peaks_array[0], peaks_array[1], "x")
    ax.grid()
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Power [dBW]')
    ax.set_xlim(0.5*10**5, 7*10**5)
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0,0))
    ax.set_title(f'PSD, cycle {s.cycle}, path {s.emitter}-{s.receiver}, {s.frequency} kHz')
    file_path = PLOT_DIR.joinpath(f'FFT+PSD_emitter_{s.emitter}_receiver_{s.receiver}_frequency_{s.frequency}_kHz.png')
    fig.savefig(file_path, dpi=500, bbox_inches='tight')
    fig.clear()

def plot_psd_and_peaks(s: Signal, bin_width: int | float, lower_bound: int = -97, upper_bound: int = -50, distance: Optional[int] = None):
    x_psd_welch: np.ndarray
    y_psd_welch: np.ndarray
    psd_welch_peaks_location: np.ndarray

    fs = s.sample_frequency
    nperseg = int(fs/bin_width)

    x_psd_welch, y_psd_welch = welch(s.x, fs, nperseg=nperseg)

    y_psd_welch_magnitude: np.ndarray = np.abs(y_psd_welch)
    y_psd_welch_magnitude_dB: np.ndarray = 10*np.log10(y_psd_welch_magnitude)

    psd_welch_peaks_location, _ = find_peaks(y_psd_welch_magnitude_dB, height=(lower_bound, upper_bound), distance=distance)
    x_psd_welch_peaks: np.ndarray = x_psd_welch[psd_welch_peaks_location].flatten()
    y_psd_welch_peaks: np.ndarray = y_psd_welch_magnitude_dB[psd_welch_peaks_location].flatten()
    peaks_array = np.array((psd_welch_peaks_location, y_psd_welch_peaks))
    print(f'{s} has peaks of magnitude {y_psd_welch_peaks} at locations {psd_welch_peaks_location}')

    PLOT_DIR = ROOT_DIR.joinpath('plots')
    if not pathlib.Path.exists(PLOT_DIR):
        pathlib.Path.mkdir(PLOT_DIR)
    fig, ax = plt.subplots(1, 1, figsize=(18, 10))
    ax.plot(x_psd_welch, y_psd_welch_magnitude_dB)
    ax.plot(peaks_array[0], peaks_array[1], "x")
    ax.grid()
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Power [dBW]')
    ax.set_xlim(0.5*10**5, 7*10**5)
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0,0))
    cycle, emitter, receiver, frequency = s.cycle, s.emitter, s.receiver, s.frequency
    ax.set_title(f'PSD, cycle {cycle}, path {emitter}-{receiver}, {frequency} kHz')
    file_path = PLOT_DIR.joinpath(f'FFT+PSD_emitter_{emitter}_receiver_{receiver}_frequency_{frequency}_kHz.png')
    fig.savefig(file_path, dpi=500, bbox_inches='tight')
    fig.clear()
