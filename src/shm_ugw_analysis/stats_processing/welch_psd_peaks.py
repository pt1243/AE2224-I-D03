import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from scipy.signal import welch, find_peaks, coherence, butter, freqs, sosfilt
from scipy.ndimage import gaussian_filter1d
from scipy.fft import fft, fftfreq, rfft, rfftfreq
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
from ..data_io.paths import ROOT_DIR, PLOT_DIR
import pathlib
from matplotlib.ticker import ScalarFormatter
from typing import Optional, Iterable, Sequence


if not pathlib.Path.exists(PLOT_DIR):
    pathlib.Path.mkdir(PLOT_DIR)


# Note that Welch's method is a cross power spectral density approximation Pxy 
# although it would be applied with respect to itself in the context of a 1D array.

# We may apply scipy.signal.coherence to create a unitless comparison of similarity
# between 2 time series vectors x and y computed using Welch cross power spectral
# density equations: Cxy = abs(Pxy)**2/(Pxx*Pyy)

# IDEALLY SHOULD ONLY BE PSD PLOTTING FUNCTION WITH MAX/MIN & SMOOTHING

def butter_lowpass(s: Signal, fs):
    sos = butter(10, 450000, btype='low', fs=fs, output='sos')
    filtered = sosfilt(sos, s)
    return filtered

def our_fft(x, fs):
    # DON'T STANDARDIZE FOR COMPARISON'S SAKE

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
    y_fft = rfft(x)
    x_fft = rfftfreq(N, d=1/fs)

    y_fft_magnitude = np.abs(y_fft)
    y_fft_magnitude_gaussian = gaussian_filter1d(y_fft_magnitude, sigma=1)

    y_fft_magnitude_dB = 10*np.log10(y_fft_magnitude)
    y_fft_magnitude_dB_gaussian = gaussian_filter1d(y_fft_magnitude_dB, sigma=5)
    array_out = np.array((x_fft, y_fft_magnitude_dB_gaussian))
    return array_out

def psd_welch(s: Signal, bin_width: int | float):
    """Calculate the PSD for a single signal."""
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
    print(f'{s} has peaks of magnitude {y_psd_welch_peaks} at locations {psd_welch_peaks_location}, corresponding to {x_psd_welch_peaks}')

    return peaks_array

def plot_signal_collection_fft(sc: Iterable[Signal], bin_width: int | float, file_label: str = None) -> None:
    """Plot and save the Welch PSD peaks for a signal collection."""
    PLOT_DIR = ROOT_DIR.joinpath('plots')
    if not pathlib.Path.exists(PLOT_DIR):
        pathlib.Path.mkdir(PLOT_DIR)
    fig, ax = plt.subplots(1, 1, figsize=(50, 10))
    for s in sc:
        print(f'Now plotting {s}')
        out_fft = our_fft(s.x, s.fs)  # compute the PSD
        ax.plot(out_fft[0], out_fft[1], label=f'Cycle {s.cycle}, {s.signal_type} {s.emitter}-{s.receiver}, {s.frequency} kHz')  # plot the PSD
    ax.grid()
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Power [dBW]')
    #ax.set_xlim(0, 2500000)
    #ax.set_xlim(0.5*10**5, 10*10**5)
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0,0))
    ax.set_title(f'PSD with peaks')
    ax.legend()
    filename = 'PSD.png' if file_label is None else f'PSD_{file_label}.png'
    file_path = PLOT_DIR.joinpath(filename)
    fig.savefig(file_path, dpi=500, bbox_inches='tight')
    print(f'Figure saved to {PLOT_DIR.joinpath(filename)}')
    plt.show()
    plt.close()
    return


def plot_signal_collection_psd_peaks(sc: Iterable[Signal], bin_width: int | float, file_label: str = None) -> None:
    """Plot and save the Welch PSD peaks for a signal collection."""
    PLOT_DIR = ROOT_DIR.joinpath('plots')
    if not pathlib.Path.exists(PLOT_DIR):
        pathlib.Path.mkdir(PLOT_DIR)
    fig, ax = plt.subplots(1, 1, figsize=(50, 10))
    for s in sc:
        print(f'Now plotting {s}')
        psd = psd_welch(s, bin_width)  # compute the PSD
        peaks_array = find_psd_peaks(s, psd)  # compute the peak locations
        ax.plot(psd[0], 10*np.log10(psd[1]),
                label=f'Cycle {s.cycle}, {s.signal_type} {s.emitter}-{s.receiver}, {s.frequency} kHz')  # plot the PSD
        ax.plot(peaks_array[0], peaks_array[1], "x")  # mark the peaks
    print(f'FINAL FREQUENCY: {psd[0][-1]}')
    ax.grid()
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Power [dBW]')
    ax.set_xlim(0, 2500000)
    #ax.set_xlim(0.5*10**5, 10*10**5)
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0,0))
    ax.set_title(f'PSD with peaks')
    ax.legend()
    filename = 'PSD.png' if file_label is None else f'PSD_{file_label}.png'
    file_path = PLOT_DIR.joinpath(filename)
    fig.savefig(file_path, dpi=500, bbox_inches='tight')
    print(f'Figure saved to {PLOT_DIR.joinpath(filename)}')
    plt.show()
    plt.close()
    return


def plot_signal_psd_peaks(
        s: Signal,
        bin_width: int | float,
        ax: Axes,
        lower_bound: int = -97,
        upper_bound: int = -50,
        distance: Optional[int] = None,
):
    """Plot a specific signal's PSD onto a given figure."""
    x_psd_welch: np.ndarray
    y_psd_welch: np.ndarray
    psd_welch_peaks_location: np.ndarray

    fs = s.sample_frequency
    nperseg = int(fs/bin_width)
    print(f'{nperseg = }')

    x_psd_welch, y_psd_welch = welch(s.x, fs, nperseg=nperseg)

    y_psd_welch_magnitude: np.ndarray = np.abs(y_psd_welch)
    y_psd_welch_magnitude_dB: np.ndarray = 10*np.log10(y_psd_welch_magnitude)

    psd_welch_peaks_location, _ = find_peaks(y_psd_welch_magnitude_dB, height=(lower_bound, upper_bound), distance=distance)
    x_psd_welch_peaks: np.ndarray = x_psd_welch[psd_welch_peaks_location].flatten()
    y_psd_welch_peaks: np.ndarray = y_psd_welch_magnitude_dB[psd_welch_peaks_location].flatten()
    peaks_array = np.array((psd_welch_peaks_location, y_psd_welch_peaks))
    print(f'{s} has peaks of magnitude {y_psd_welch_peaks} at locations {psd_welch_peaks_location}, corresponding to {x_psd_welch_peaks}')

    ax.plot(x_psd_welch, y_psd_welch_magnitude_dB)
    ax.plot(peaks_array[0], peaks_array[1], "x")
    return


def get_baseline(s: Signal):
    """Gets the baseline (cycle 0) for a given signal."""
    return Signal('0', s.signal_type, s.emitter, s.receiver, s.frequency)


def calculate_coherence(s: Signal, bin_width: float | int):
    """Calculate the coherence between a given signal and baseline."""
    baseline = get_baseline(s)
    baseline_psd = psd_welch(baseline, bin_width=bin_width)
    s_psd = psd_welch(s, bin_width=bin_width)
    coherence_arr = coherence(s_psd[1], baseline_psd[1])
    return coherence_arr


def plot_coherence(sc: Iterable[Signal], bin_width: float | int, sigma: float | int):
    """Plot the coherence for a collection of signals."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    for s in sc:
        coherence_arr = calculate_coherence(s, bin_width=bin_width)
        smoothed = gaussian_filter1d(coherence_arr[1], sigma=sigma)
        ax.plot(coherence_arr[0], smoothed, 
                label=f'Cycle {s.cycle}, {s.signal_type}, {s.emitter}-{s.receiver}, {s.frequency} kHz')
    ax.legend()
    ax.grid()
    # ax.set_yscale('log')
    filepath = PLOT_DIR.joinpath('coherence.png')
    plt.savefig(filepath, dpi=500, bbox_inches='tight')
    print(f'Coherence plot saved to {PLOT_DIR.joinpath(filepath)}')
    return

def plot_coherence_3d(sc: Iterable[Signal], bin_width: float | int):
    """Plot the coherence for a collection of signals."""
    ax = plt.axes(projection='3d')
    for s in sc:
        coherence_arr = calculate_coherence(s, bin_width=bin_width)
        smoothed = gaussian_filter1d(coherence_arr[1], sigma=1)
        
        X = coherence_arr[0]*2500000*2
        Y = smoothed
        Z = np.ones((X.size, Y.size)) * int(s.cycle)
        ax.plot_wireframe(X, Y, Z)
        #ax.plot_surface(X, Y, Z)
    
    ax.legend()
    #ax.grid()
    ax.set_xlabel('response frequency bands')
    ax.set_ylabel('coherence')
    ax.view_init(vertical_axis = 'y')
    # elev = -52, azim = 90, roll = 45
    #ax.azim = 90
    #ax.elev = -52
    ax.set_zlabel('cycles')
    # ax.set_yscale('log')
    ax.set_title(f'Cycle {s.cycle}, {s.signal_type}, {s.emitter}-{s.receiver}, {s.frequency} kHz')
    filepath = PLOT_DIR.joinpath('coherence.png')
    plt.savefig(filepath, dpi=500)
    plt.show()
    print(f'Coherence plot saved to {PLOT_DIR.joinpath(filepath)}')
    return

'''
def coherence(x_freq_baseline, x_freq_all_but_baseline):
    f, Cxy = coherence(x_freq_baseline, x_freq_all_but_baseline)
    return np.array([f, Cxy])
x_psd_welch, y_psd_welch = welch(s.x, fs, nperseg=nperseg)

psd_welch()

for i in allowed_cycles and not allowed_cycles = '0':
    coherence_array = coherence(y_psd_welch_magnitude_dB[0], y_psd_welch_magnitude_dB[i])

ax = plt.axes(projection='3d')
X = coherence_array[0]
Y = coherence_array[1]
Z: X, Y at different cycles 
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_title('surface')
'''