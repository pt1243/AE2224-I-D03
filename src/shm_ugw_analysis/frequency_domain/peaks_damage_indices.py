import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from ..data_io.load_signals import (
    Signal,
    signal_collection,
    path_collection,
    frequency_collection,
    InvalidSignalError,
    allowed_cycles,
    allowed_signal_types,
    allowed_emitters,
    allowed_receivers,
    allowed_frequencies,
    relevant_cycles,
    all_paths,
)
from ..data_io.paths import PLOT_DIR
from .welch_psd_peaks import our_fft, butter_lowpass, real_find_peaks
from typing import Literal, Optional
import numpy as np


Optimum = Literal["minimum", "maximum"]

local_optima_bounds: dict[int, dict[Optimum, dict[int, tuple[int | float, int | float] | None]]] = {
    100: {
        "maximum": {
            1: (95, 105),
            2: (180, 190),
            3: (285, 295),
        },
        "minimum": {
            1: (0, 15),
            2: (37.5, 45),
            3: (155, 165),
            4: (215, 225),
        },
    },
    120: {
        "maximum": {
            1: (115, 130),
            2: (220, 230),
            3: (345, 350),
        },
        "minimum": {
            1: (0, 15),
            2: (42.5, 52.5),
            3: (185, 200),
            4: (250, 265),
        }
    },
    140: {
        "maximum": {
            1: (135, 145),
            2: (245, 260),
            3: (380, 410),
        },
        "minimum": {
            1: (5, 15),
            2: (47.5, 57.5),
            3: (215, 230),
            4: (275, 290),
        }
    },
    160: {
        "maximum": {
            1: (150, 180),
            2: (265, 285),
            3: (420, 430),
        },
        "minimum": {
            1: (5, 20),
            2: (50, 65),
            3: (230, 250),
            4: (300, 320),
        }
    },
    180: {
        "maximum": {
            1: (165, 185),
            2: (275, 290),
            3: (440, 455),
        },
        "minimum": {
            1: None,
            2: (50, 72.5),
            3: (230, 260),
            4: (330, 345),
        }
    }
}


def search_peaks_arrays(peaks_frequencies: np.ndarray, peaks_y: np.ndarray, lower: int | float, upper: int | float):
    """Search the peaks arrays for the given optima location and return the magnitude."""
    # print(f'{peaks_frequencies = }')
    # print(f'{peaks_y = }')
    # convert to kHz
    lower *= 1000
    upper *= 1000
    # print(f'{lower = }, {upper = }')
    # Locate maxima
    index = np.argwhere((lower <= peaks_frequencies ) & (peaks_frequencies <= upper))[0][0]
    return peaks_y[index]


def generate_magnitude_array(optimum_type: Optimum, optimum_number: int, f: int, use_dB: bool = True):
    """For a given excitation frequency and optima, generate the arrays to be plotted of cycle numbers and the maxima
    as they vary per cycle.
    """
    labels_arr = []
    optima_arr = []

    for cycle in relevant_cycles:
        optimum_key = local_optima_bounds[f][optimum_type][optimum_number]

        # skip optima which are not well behaved
        if optimum_key is None:
            continue
        # unpack bounds
        lower, upper = optimum_key

        # average over paths
        fc = frequency_collection((cycle,), ('received',), f, paths=None)
        for i, s in enumerate(fc):
            x = s.x
            fs = s.sample_frequency
            fft = our_fft(x, fs, sigma=20)
            if i == 0:
                average_fft = fft
            else:
                average_fft += fft
        average_fft /= i + 1

        magnitude = search_peaks_arrays(average_fft[0], average_fft[1], lower, upper)
        if not use_dB:
            magnitude = 10**(magnitude/10)

        labels_arr.append(cycle)
        optima_arr.append(magnitude)
    
    return labels_arr, optima_arr


def plot_DI(optimum_type: Optimum, optimum_number: int, use_dB: bool = True, ax: Optional[Axes] = None):
    """Plot the damage index for a given optima type and location (eg. first minima)."""
    linestyles = {
        100: 'solid',
        120: 'dotted',
        140: 'dashed',
        160: 'dashdot',
        180: (5, (10, 3)),
    }
    label_mapping = {
        0: '0',
        1: '1',
        2: '1,000',
        3: '10,000',
        4: '20,000',
        5: '30,000',
        6: '40,000',
        7: '50,000',
        8: '60,000',
        9: '70,000',
    }
    show_only_subplot = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        show_only_subplot = True
    def handle_tick_errors(x, pos):
        try:
            return label_mapping[pos]
        except KeyError:
            return
    ax.xaxis.set_major_formatter(handle_tick_errors)
    for f in allowed_frequencies:
        labels_arr, optima_arr = generate_magnitude_array(optimum_type, optimum_number, f, use_dB)
        ax.plot(labels_arr, optima_arr, label=f'{f} kHz, averaged over all paths', linestyle=linestyles[f])
    ax.set_xlabel(f'Cycle')
    if use_dB:
        ax.set_ylabel(f'Magnitude [dBV]')
    else:
        ax.set_ylabel(f'Magnitude [V]')
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha('right')
    if show_only_subplot:
        ax.legend()
        filepath = PLOT_DIR / f'DI_{optimum_type}_{optimum_number}.png'
        plt.savefig(filepath, dpi=500, bbox_inches='tight')
        # plt.show()
    else:
        ax.set_title(f'{optimum_type.title()} {optimum_number}')
    return


def plot_all_DIs(use_dB: bool = True):
    """Plot all optima."""
    fig, axs = plt.subplots(4, 2, figsize=(12, 18))
    for i in range(3):
        plot_DI("maximum", i+1, use_dB, axs[i, 0])
    for i in range(4):
        plot_DI("minimum", i+1, use_dB, axs[i, 1])
    axs[-1, 0].axis('off')
    fig.subplots_adjust(hspace=0.4, wspace=0.2)
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower left', bbox_to_anchor=(0.1, 0.1))
    filepath = PLOT_DIR.joinpath(f"FFT Peaks all DIs, db = {use_dB}")
    plt.savefig(filepath, dpi=500, bbox_inches='tight')
    # plt.show()
