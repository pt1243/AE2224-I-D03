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

#plot_signal_collection_psd_peaks(sc, bin_width=2000, file_label='changing_cycles')

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

#plot_coherence(sc_coherence, bin_width=5000)
#plot_coherence_3d(sc_coherence, bin_width=3000)

fig, ax = plt.subplots(1, 1, figsize=(12, 10))

f = 180
for cycle in relevant_cycles:
    fc = frequency_collection(cycles=(cycle,), signal_types=('received',), frequency=f, paths=None, residual=False)
    for i, s in enumerate(fc):
        fs = s.sample_frequency
        buttered_array = butter_lowpass(s.x, fs, order=20)
        #buttered_array = s.x
        buttered_fft = our_fft(buttered_array, fs, sigma=20)
        if i == 0:
            average_buttered_fft = buttered_fft
        else:
            average_buttered_fft += buttered_fft
    average_buttered_fft /= (i + 1)
    x_peaks, y_peaks = real_find_peaks(average_buttered_fft)
    x_min, y_min = real_find_peaks((average_buttered_fft[0], -average_buttered_fft[1]))
    ax.plot(x_peaks, y_peaks, "x")
    ax.plot(x_min, -y_min, "x")
    ax.plot(average_buttered_fft[0], average_buttered_fft[1], label=f'buttered cycle {cycle}, received, all paths, {f} kHz')
    #unbuttered_fft = our_fft(s.x, fs)
    #plt.plot(unbuttered_fft[0], unbuttered_fft[1], label=f'unbuttered Cycle {s.cycle}, {s.signal_type}, {s.emitter}-{s.receiver}, {s.frequency}')



ax.legend()
ax.set_xlim(0, 450000)
plt.show()

# plot_signal_collection_psd_peaks(sc_4, bin_width=4000, file_label='residuals')
