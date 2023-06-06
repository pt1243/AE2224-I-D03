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

from shm_ugw_analysis.stats_processing.welch_psd_peaks import plot_signal_collection_psd_peaks, calculate_coherence, plot_coherence, plot_coherence_3d, butter_lowpass, our_fft, plot_signal_collection_fft, real_find_peaks

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

############################################
### VARYING CYCLES, CONSTANT FREQUENCIES ###
############################################

for f in allowed_frequencies:
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    # 12501 full freq range
    envelope_matrix = np.zeros([len(relevant_cycles), 2250])
    for row_index, cycle in enumerate(relevant_cycles):
        fc = frequency_collection(cycles=(cycle,), signal_types=('received',), frequency=f, paths=None, residual=True)
        for i, s in enumerate(fc):
            fs = s.sample_frequency
            fft_array = our_fft(s.x, fs, sigma=20)
            if i == 0:
                average_fft_array = fft_array
            else:
                average_fft_array += fft_array
        average_fft_array /= (i + 1)
        ### PLOTTING VALUES LESS THAN 450 KHZ ONLY ###
        domain_restriction_index = np.where(average_fft_array[0] <= 450000)[-1][-1]
        average_fft_array = np.array([average_fft_array[0][0:domain_restriction_index], average_fft_array[1][0:domain_restriction_index]])
        ### PEAK FINDING & PLOTTING ###
        x_peaks, y_peaks = real_find_peaks(average_fft_array)
        x_min, y_min = real_find_peaks((average_fft_array[0], -average_fft_array[1]))
        #ax.plot(x_peaks, y_peaks, "x")
        #ax.plot(x_min, -y_min, "x")

        ### CURVE PLOTTING ###
        #ax.plot(average_fft_array[0], average_fft_array[1], label=f'received, all paths, cycle {cycle}')
        envelope_matrix[row_index, :] = average_fft_array[1]
    print(f'Rendering plots for excitation frequency {f}...')

    ### ENVELOPE ###
    upper_envelope = np.max(envelope_matrix, axis=0)
    lower_envelope = np.min(envelope_matrix[1:, :], axis=0)
    ax.plot(average_fft_array[0], upper_envelope, color='#030aa7', label='Upper envelope', linestyle='dashed')
    ax.plot(average_fft_array[0], lower_envelope, color='#00ff00', label='Lower envelope', linestyle='dashdot')
    # Entire Range:
    #ax.fill_between(average_fft_array[0], lower_envelope, upper_envelope, color='#8cffdb')
    
    ### LAYERS WITHIN ENVELOPE ###
    # relevant_cycles: Final = ('0', '1', '1000', '10000', '20000', '30000', '40000', '50000', '60000', '70000')
    # Layer 1: 0, 1, 1000
    # Layer 2: 10k, 20k, 30k
    # Layer 3: 40k, 50k
    # Layer 4: 60k, 70k

    # New Gradient: gradientRed2yellow
    # color='#e5f614'
    # color='#f8cb0f'
    # color='#fa8b0a'
    # color='#ff0000'

    # Old Gradient: Yellows
    # color='#f0ff00'
    # color='#ffe700'
    # color='#ffdb00'
    # color='#ffce00'

    ax.fill_between(average_fft_array[0], envelope_matrix[0, :], envelope_matrix[2, :], color='#e5f614', label='Layer 1: 0 - 1,000 cycles')
    ax.fill_between(average_fft_array[0], envelope_matrix[2, :], envelope_matrix[5, :], color='#f8cb0f', label='Layer 2: 10,000 - 30,000 cycles')
    ax.fill_between(average_fft_array[0], envelope_matrix[5, :], envelope_matrix[7, :], color='#fa8b0a', label='Layer 3: 40,000 - 50,000 cycles')
    ax.fill_between(average_fft_array[0], envelope_matrix[7, :], envelope_matrix[9, :], color='#ff0000', label='Layer 4: 60,000 - 70,000 cycles')

    ### PLOTTING PARAMETERS ###
    ax.legend()
    #ax.set_title(f'FFT - Excitation Frequency {f} kHz, Received, All Paths (Averaged), All Cycles, residual=True')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Magnitude [dBV]')
    ax.set_xlim(0, 450_000)
    ax.set_ylim(-13, 6)
    file_path = os.path.join(PLOT_DIR, f'FFT - Excitation Frequency {f} kHz, Received, All Paths (Averaged), All Cycles, residual=True')
    plt.savefig(file_path, dpi=500, bbox_inches='tight')
    #plt.show()
    #plt.clf()
    #plt.close()

############################################
### VARYING FREQUENCIES, CONSTANT CYCLES ###
############################################

for cycle in relevant_cycles:
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    # col = 25001
    # 12501 full freq range
    envelope_matrix = np.empty([len(allowed_frequencies), 2250])
    for row_index, frequency in enumerate(allowed_frequencies):
        sc = signal_collection(cycles=(cycle,), signal_types=('received',), frequencies=(frequency,), emitters=allowed_emitters, receivers=allowed_receivers, residual=True)
        for i, s in enumerate(sc):
            fs = s.sample_frequency
            fft_array = our_fft(s.x, fs, sigma=20)
            if i == 0:
                average_fft_array = fft_array
            else:
                average_fft_array += fft_array
        average_fft_array /= (i + 1)
        ### PLOTTING VALUES LESS THAN 450 KHZ ONLY ###
        domain_restriction_index = np.where(average_fft_array[0] <= 450000)[-1][-1]
        average_fft_array = np.array([average_fft_array[0][0:domain_restriction_index], average_fft_array[1][0:domain_restriction_index]])
        ### PEAK FINDING & PLOTTING ###
        x_peaks, y_peaks = real_find_peaks(average_fft_array)
        x_min, y_min = real_find_peaks((average_fft_array[0], -average_fft_array[1]))
        #ax.plot(x_peaks, y_peaks, "x")
        #ax.plot(x_min, -y_min, "x")

        ### CURVE PLOTTING ###
        #ax.plot(average_fft_array[0], average_fft_array[1], label=f'received, all paths, {f} kHz')
        envelope_matrix[row_index, :] = average_fft_array[1]
    print(f'Rendering plots for cycle {cycle}...')

    ### ENVELOPE ###
    upper_envelope = np.max(envelope_matrix, axis=0)
    lower_envelope = np.min(envelope_matrix, axis=0)
    ax.plot(average_fft_array[0], upper_envelope, color='#030aa7', label='Upper envelope', linestyle='dashed')
    ax.plot(average_fft_array[0], lower_envelope, color='#00ff00', label='Lower envelope', linestyle='dashdot')
    # Entire Range:
    #ax.fill_between(average_fft_array[0], lower_envelope, upper_envelope, color='#8cffdb')

    ### LAYERS WITHIN ENVELOPE ###
    # Layer 1: 100-120 kHz
    # Layer 2: 120-140 kHz
    # Layer 3: 140-160 kHz
    # Layer 4: 160-180 kHz

    ax.fill_between(average_fft_array[0], envelope_matrix[0, :], envelope_matrix[1, :], color='#e5f614', label='Layer 1: 100 - 120 kHz')
    ax.fill_between(average_fft_array[0], envelope_matrix[1, :], envelope_matrix[2, :], color='#f8cb0f', label='Layer 2: 120 - 140 kHz')
    ax.fill_between(average_fft_array[0], envelope_matrix[2, :], envelope_matrix[3, :], color='#fa8b0a', label='Layer 3: 140 - 160 kHz')
    ax.fill_between(average_fft_array[0], envelope_matrix[3, :], envelope_matrix[4, :], color='#ff0000', label='Layer 4: 160 - 180 kHz')
    
    ### PLOTTING PARAMETERS ###
    ax.legend()
    #ax.set_title(f'FFT - Cycle {cycle}, Received, All Paths (Averaged), All Excitation Frequencies, residual=True')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Magnitude [dBV]')
    file_path = os.path.join(PLOT_DIR, f'FFT - Cycle {cycle}, Received, All Paths (Averaged), All Excitation Frequencies, residual=True')
    plt.savefig(file_path, dpi=500, bbox_inches='tight')
    #plt.show()
    #plt.clf()
    #plt.close()