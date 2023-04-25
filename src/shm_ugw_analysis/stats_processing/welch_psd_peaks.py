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


# Note that Welch's method is a cross power spectral density approximation Pxy 
# although it would be applied with respect to itself in the context of a 1D array.

# We may apply scipy.signal.coherence to create a unitless comparison of similarity
# between 2 time series vectors x and y computed using Welch cross power spectral
# density equations: Cxy = abs(Pxy)**2/(Pxx*Pyy)

# IDEALLY SHOULD ONLY BE PSD PLOTTING FUNCTION WITH MAX/MIN & SMOOTHING


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
    print(f'{s} has peaks of magnitude {y_psd_welch_peaks} at locations {psd_welch_peaks_location}, corresponding to {x_psd_welch_peaks}')

    return peaks_array


def plot_signal_collection_psd_peaks(sc: Iterable[Signal], bin_width: int | float, file_label: str = None) -> None:
    """Plot and save the Welch PSD peaks for a signal collection."""
    PLOT_DIR = ROOT_DIR.joinpath('plots')
    if not pathlib.Path.exists(PLOT_DIR):
        pathlib.Path.mkdir(PLOT_DIR)
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    for s in sc:
        print(f'Now plotting {s}')
        psd = psd_welch(s, bin_width)  # compute the PSD
        peaks_array = find_psd_peaks(s, psd)  # compute the peak locations
        ax.plot(psd[0], 10*np.log10(psd[1]),
                label=f'Cycle {s.cycle}, {s.signal_type} {s.emitter}-{s.receiver}, {s.frequency} kHz')  # plot the PSD
        ax.plot(peaks_array[0], peaks_array[1], "x")  # mark the peaks
    ax.grid()
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Power [dBW]')
    ax.set_xlim(0.5*10**5, 10*10**5)
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0,0))
    ax.set_title(f'PSD with peaks')
    ax.legend()
    filename = 'PSD.png' if file_label is None else f'PSD_{file_label}.png'
    file_path = PLOT_DIR.joinpath(filename)
    fig.savefig(file_path, dpi=500, bbox_inches='tight')
    print(f'Figure saved to {PLOT_DIR.joinpath(filename)}')
    fig.clear()
    return


def plot_signal_psd_peaks(
        s: Signal,
        bin_width: int | float,
        ax: Axes,
        lower_bound: int = -97,
        upper_bound: int = -50,
        distance: Optional[int] = None,
):
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

    PLOT_DIR = ROOT_DIR.joinpath('plots')
    if not pathlib.Path.exists(PLOT_DIR):
        pathlib.Path.mkdir(PLOT_DIR)

    ax.plot(x_psd_welch, y_psd_welch_magnitude_dB)
    ax.plot(peaks_array[0], peaks_array[1], "x")
    return


def get_baseline(s: Signal):
    return Signal('0', s.signal_type, s.emitter, s.receiver, s.frequency)
