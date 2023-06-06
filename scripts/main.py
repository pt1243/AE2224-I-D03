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


# sb = Signal('0', 'received', 1, 4, 100)
# s1 = Signal('1000', 'received', 1, 4, 100)
# s2 = Signal('70000', 'received', 1, 4, 100)
# plot_psd_and_peaks(s, 4000)

# sc = signal_collection(
#    cycles=relevant_cycles,
#    signal_types=('received',),
#    emitters=(1, 2, 3),
#    receivers=(4, 5, 6),
#    frequencies=(180,),
#)


# plot_all_DIs(use_dB=True)

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


fig, ax = plt.subplots(1, 1, figsize=(8, 8))

f = 100
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


cycle = '1'

fc = frequency_collection(cycles=(cycle,), signal_types=('received',), frequency=f, paths=None, residual=False)
for i, s in enumerate(fc):
    fs = s.sample_frequency
    # buttered_array = butter_lowpass(s.x, fs, order=20)
    buttered_array = s.x
    buttered_fft = our_fft(buttered_array, fs, sigma=20)
    if i == 0:
        average_buttered_fft = buttered_fft
    else:
        average_buttered_fft += buttered_fft
average_buttered_fft /= (i + 1)
x_peaks, y_peaks = real_find_peaks(average_buttered_fft)
print(x_peaks)
print(y_peaks)
x_min, y_min = real_find_peaks((average_buttered_fft[0], -average_buttered_fft[1]))
print(x_min)
print(-y_min)

# ax.plot(x_peaks, y_peaks, "x")
# ax.plot(x_min, -y_min, "x")
ax.plot(average_buttered_fft[0], average_buttered_fft[1], label='Smoothed signal FFT')

optima = {
    "First minimum": (10599.57601696, -1.47720732, 'p'),
    "Second minimum": (41198.35206592, -3.8851639, '*'),
    "First maximum": (101995.92016319, 6.01553543, 'o'),
    "Third minimum": (161793.52825887, -5.89043866, '+'),
    "Second maximum": (187792.48830047, -0.73183618, 'v'),
    "Fourth minimum": (223191.07235711, -8.28273189, 'x'),
    "Third maximum": (290188.3924643, 0.02122642, '^'),
}

for k, v in optima.items():
    ax.plot(v[0], v[1], v[2], mew=2, ms=8, label=k)

ax.legend(loc='lower left')
ax.set_xlim(0, 450000)
ax.set_ylim(-13, 7)
ax.grid()
ax.set_xlabel('Frequency [Hz]')
ax.set_ylabel('Magnitude [dBV]')
filepath = PLOT_DIR / 'smoothed.png'
plt.savefig(filepath, dpi=500, bbox_inches='tight')
# plt.show()

# y_peaks = np.array([ 0.33249576,  2.57209416,  5.0102436 ,  7.44287459,  7.19390225, 10.77036642])

# frequency_array = np.array([ 12599.49602016,  42998.2800688 , 162393.50425983, 223391.06435743, 374985.00059998, 433182.67269309])

# mag = search_peaks_arrays(100, 'minima', 2, frequency_array, y_peaks)

# print(mag)

plt.clf()
plt.close()

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
fc = frequency_collection(cycles=(cycle,), signal_types=('received',), frequency=f, paths=None, residual=False)
for i, s in enumerate(fc):
    fs = s.sample_frequency
    # buttered_array = butter_lowpass(s.x, fs, order=20)
    buttered_array = s.x
    buttered_fft = our_fft(buttered_array, fs, sigma=1e-6)
    if i == 0:
        average_buttered_fft = buttered_fft
    else:
        average_buttered_fft += buttered_fft
average_buttered_fft /= (i + 1)
x_peaks, y_peaks = real_find_peaks(average_buttered_fft)
print(x_peaks)
print(y_peaks)
x_min, y_min = real_find_peaks((average_buttered_fft[0], -average_buttered_fft[1]))
print(x_min)
print(-y_min)

# ax.plot(x_peaks, y_peaks, "x")
# ax.plot(x_min, -y_min, "x")
ax.plot(average_buttered_fft[0], average_buttered_fft[1], label='Unsmoothed signal FFT')

# sc1 = signal_collection(
#     cycles=('30000',),
#     signal_types=('excitation',),
#     emitters=(1,),
#     receivers=(4,),
#     frequencies=(100, 120, 140, 160, 180,),
# )

# plot_signal_collection_psd_peaks(sc1, bin_width=2000, file_label='excitation_changing_frequencies')

# sc2 = signal_collection(
#     cycles=relevant_cycles,
#     signal_types=('excitation',),
#     emitters=(1,),
#     receivers=(4,),
#     frequencies=(180,),
# )

# plot_signal_collection_psd_peaks(sc2, bin_width=2000, file_label='excitation_changing_cycles')


# sc3 = signal_collection(
#     cycles=('30000',),
#     signal_types=('received',),
#     emitters=(1,),
#     receivers=(4,),
#     frequencies=(100, 120, 140, 160, 180,),
# )

# plot_signal_collection_psd_peaks(sc3, bin_width=2000, file_label='receieved_changing_frequencies')

# sc4 = signal_collection(
#     cycles=relevant_cycles,
#     signal_types=('received',),
#     emitters=(1,),
#     receivers=(4,),
#     frequencies=(180,),
# )

# plot_signal_collection_psd_peaks(sc4, bin_width=2000, file_label='received_changing_cycles')


# sc_coherence = signal_collection(
#     cycles=relevant_cycles,
#     signal_types=('excitation',),
#     emitters=(1,),
#     receivers=(4,),
#     frequencies=(180,)
# )

# plot_coherence(sc_coherence, bin_width=4000, sigma=2)


sc_4 = signal_collection(
    cycles=relevant_cycles,
    signal_types=('received',),
    emitters=(2,),
    receivers=(5,),
    frequencies=(100,)
)

ax.legend(loc='lower left')
ax.set_xlim(0, 450000)

ax.set_ylim(-13, 7)
ax.grid()
ax.set_xlabel('Frequency [Hz]')
ax.set_ylabel('Magnitude [dBV]')
filepath = PLOT_DIR / 'unsmoothed.png'
plt.savefig(filepath, dpi=500, bbox_inches='tight')
plt.show()

# plot_signal_collection_psd_peaks(sc_4, bin_width=4000, file_label='residuals')
