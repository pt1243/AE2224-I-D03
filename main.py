import matplotlib.pyplot as plt
import numpy as np
from shm_ugw_analysis.data_io.load_signals import (
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
from shm_ugw_analysis.data_io.paths import PLOT_DIR

from shm_ugw_analysis.stats_processing.welch_psd_peaks import plot_signal_collection_psd_peaks, calculate_coherence, plot_coherence, plot_coherence_3d, butter_lowpass, our_fft, plot_signal_collection_fft, real_find_peaks

from shm_ugw_analysis.stats_processing.peaks_damage_indices import search_peaks_arrays, generate_magnitude_array, plot_DI, plot_all_DIs

plot_all_DIs(use_dB=True)

# fc = frequency_collection(('0'), ('received',), 100, paths=None)
# for i, s in enumerate(fc):
#     fs = s.sample_frequency
#     buttered_array = butter_lowpass(s.x, fs, order=20)
#     # buttered_array = s.x
#     buttered_fft = our_fft(buttered_array, fs, sigma=20)
#     if i == 0:
#         average_buttered_fft = buttered_fft
#     else:
#         average_buttered_fft += buttered_fft
# average_buttered_fft /= (i + 1)
# x_peaks, y_peaks = real_find_peaks(average_buttered_fft)
# x_min, y_min = real_find_peaks((average_buttered_fft[0], -average_buttered_fft[1]))

# print(repr(x_min))
# print(repr(y_min))


# fig, ax = plt.subplots(1, 1, figsize=(12, 10))

# f = 100
# for cycle in relevant_cycles:
#     fc = frequency_collection(cycles=(cycle,), signal_types=('received',), frequency=f, paths=None, residual=False)
#     for i, s in enumerate(fc):
#         fs = s.sample_frequency
#         buttered_array = butter_lowpass(s.x, fs, order=20)
#         # buttered_array = s.x
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

# y_peaks = np.array([ 0.33249576,  2.57209416,  5.0102436 ,  7.44287459,  7.19390225, 10.77036642])

# frequency_array = np.array([ 12599.49602016,  42998.2800688 , 162393.50425983, 223391.06435743, 374985.00059998, 433182.67269309])

# mag = search_peaks_arrays(100, 'minima', 2, frequency_array, y_peaks)

# print(mag)