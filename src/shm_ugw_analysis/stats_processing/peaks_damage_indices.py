import matplotlib.pyplot as plt
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
from .welch_psd_peaks import our_fft, butter_lowpass, real_find_peaks
from typing import Literal
import numpy as np


local_optima_bounds: dict[int, dict[str, dict[int, tuple[int, int]]]] = {
    100: {
        "maxima": {
            1: (95, 105),
            2: (180, 190),
            3: (285, 295),
        },
        "minima": {
            1: (0, 15),
            2: (37.5, 45),
            3: (155, 165),
            4: (215, 225),
        },
    },
    120: {
        "maxima": {
            1: (115, 130),
            2: (220, 230),
            3: (345, 350),
        },
        "minima": {
            1: (0, 15),
            2: (42.5, 52.5),
            3: (185, 200),
            4: (250, 265),
        }
    },
    140: {
        "maxima": {
            1: (135, 145),
            2: (245, 260),
            3: (380, 410),
        },
        "minima": {
            1: (5, 15),
            2: (47.5, 57.5),
            3: (215, 230),
            4: (275, 290),
        }
    },
    160: {
        "maxima": {
            1: (150, 180),
            2: (265, 285),
            3: (420, 430),
        },
        "minima": {
            1: (5, 20),
            2: (50, 65),
            3: (230, 250),
            4: (300, 320),
        }
    },
    180: {
        "maxima": {
            1: (165, 185),
            2: (275, 290),
            3: (440, 455),
        },
        "minima": {
            1: None,
            2: (50, 72.5),
            3: (230, 260),
            4: (330, 345),
        }
    }
}


# local_optima_bounds[100]["maxima"][1] -> (lower, upper) or None if not well behaved


def search_peaks_arrays(peaks_frequencies: np.ndarray, peaks_y: np.ndarray, lower: int | float, upper: int | float):
    """Search the peaks arrays for the given optima location and return the magnitude."""
    # print(f'{peaks_frequencies = }')
    # print(f'{peaks_y = }')
    lower *= 1000
    upper *= 1000
    # print(f'{lower = }, {upper = }')
    index = np.argwhere((lower <= peaks_frequencies ) & (peaks_frequencies <= upper))[0][0]
    return peaks_y[index]


def plot_DI(optima_type: Literal["maxima", "minima"], optima_number: int):
    """Plot the damage index for a given optima type and location (eg. first minima)."""
    
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))


# fig, ax = plt.subplots(1, 1, figsize=(12, 10))

# f = 180
# for cycle in relevant_cycles:
#     fc = frequency_collection(cycles=(cycle,), signal_types=('received',), frequency=f, paths=None, residual=False)
#     for i, s in enumerate(fc):
#         fs = s.sample_frequency
#         buttered_array = butter_lowpass(s.x, fs, order=20)
#         #buttered_array = s.x
#         buttered_fft = our_fft(buttered_array, fs, sigma=20)
#         if i == 0:
#             average_buttered_fft = buttered_fft
#         else:
#             average_buttered_fft += buttered_fft
#     average_buttered_fft /= (i + 1)
#     x_peaks, y_peaks = real_find_peaks(average_buttered_fft)
#     x_min, y_min = real_find_peaks((average_buttered_fft[0], -average_buttered_fft[1]))
#     ax.plot(x_peaks, y_peaks, "x")
#     ax.plot(x_min, -y_min, "x")
#     ax.plot(average_buttered_fft[0], average_buttered_fft[1], label=f'buttered cycle {cycle}, received, all paths, {f} kHz')
#     #unbuttered_fft = our_fft(s.x, fs)
#     #plt.plot(unbuttered_fft[0], unbuttered_fft[1], label=f'unbuttered Cycle {s.cycle}, {s.signal_type}, {s.emitter}-{s.receiver}, {s.frequency}')



# ax.legend()
# ax.set_xlim(0, 450000)
# plt.show()