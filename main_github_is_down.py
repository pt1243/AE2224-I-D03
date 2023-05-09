import matplotlib.pyplot as plt
import numpy as np
import os
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

<<<<<<< Updated upstream
from shm_ugw_analysis.stats_processing.welch_psd_peaks import plot_signal_collection_psd_peaks, calculate_coherence, plot_coherence, butter_lowpass, our_fft, plot_signal_collection_fft, real_find_peaks
=======
from shm_ugw_analysis.stats_processing.welch_psd_peaks import plot_signal_collection_psd_peaks, calculate_coherence, plot_coherence, plot_coherence_3d, butter_lowpass, our_fft, plot_signal_collection_fft, real_find_peaks
>>>>>>> Stashed changes

sb = Signal('0', 'received', 1, 4, 100)
s1 = Signal('1000', 'received', 1, 4, 100)
s2 = Signal('70000', 'received', 1, 4, 100)
# plot_psd_and_peaks(s, 4000)

sc = signal_collection(
    cycles=relevant_cycles,
    signal_types=('received',),
    emitters=(1, 2, 3),
    receivers=(4, 5, 6),
    frequencies=(180,),
)

#plot_signal_collection_psd_peaks(sc, bin_width=2000, file_label='changing_cycles')

sc_coherence = signal_collection(
    cycles=relevant_cycles,
    signal_types=('received',),
    emitters=(2,),
    receivers=(5,),
    frequencies=(100,)
)

#plot_coherence(sc_coherence, bin_width=5000)
#plot_coherence_3d(sc_coherence, bin_width=3000)
'''
fig, ax = plt.subplots(1, 1, figsize=(12, 10))

f = 180
for cycle in relevant_cycles:
    fc = frequency_collection(cycles=(cycle,), signal_types=('received',), frequency=f, paths=None)
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
'''

for cycle in relevant_cycles:
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    for frequency in allowed_frequencies:
<<<<<<< Updated upstream
        sc = signal_collection(cycles=(cycle,), signal_types=('received',), frequencies=(frequency,), emitters=allowed_emitters, receivers=allowed_receivers, residual=True)
=======
        sc = signal_collection(cycles=(cycle,), signal_types=('received',), frequencies=(frequency,), emitters=allowed_emitters, receivers=allowed_receivers)
>>>>>>> Stashed changes
        for i, s in enumerate(sc):
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
        ax.plot(average_buttered_fft[0], average_buttered_fft[1], label=f'buttered cycle {cycle}, received, all paths, {frequency} kHz')
        #unbuttered_fft = our_fft(s.x, fs)
        #plt.plot(unbuttered_fft[0], unbuttered_fft[1], label=f'unbuttered Cycle {s.cycle}, {s.signal_type}, {s.emitter}-{s.receiver}, {s.frequency}')
    ax.legend()
    ax.set_xlim(0, 450000)
<<<<<<< Updated upstream
    file_path = os.path.join(PLOT_DIR, f'buttered cycle {cycle}, received, all paths (averaged), all allowed_frequencies, residual=True')
=======
    file_path = os.path.join(PLOT_DIR, f'buttered cycle {cycle}, received, all paths (averaged), all allowed_frequencies')
>>>>>>> Stashed changes
    plt.savefig(file_path, dpi = 500)
    #plt.show()
    plt.clf()