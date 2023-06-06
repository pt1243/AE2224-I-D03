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

from shm_ugw_analysis.frequency_domain.welch_psd_peaks import plot_signal_collection_psd_peaks, calculate_coherence, plot_coherence, plot_coherence_3d, butter_lowpass, our_fft, plot_signal_collection_fft, real_find_peaks

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
    envelope_matrix = np.empty([len(relevant_cycles), 12501])
    for row_index, cycle in enumerate(relevant_cycles):
        fc = frequency_collection(cycles=(cycle,), signal_types=('received',), frequency=f, paths=None, residual=False)
        for i, s in enumerate(fc):
            fs = s.sample_frequency
            ### BUTTER ###
            buttered_array = butter_lowpass(s.x, fs, order=20)
            buttered_fft = our_fft(buttered_array, fs, sigma=20)
            ### UNBUTTERED ###
            unbuttered_array = s.x
            unbuttered_fft = our_fft(unbuttered_array, fs, sigma=20)
            
            if i == 0:
                average_buttered_fft = buttered_fft
                average_unbuttered_fft = unbuttered_fft
            else:
                average_buttered_fft += buttered_fft
                average_unbuttered_fft += unbuttered_fft
        average_buttered_fft /= (i + 1)
        average_unbuttered_fft /= (i + 1)

        ### PEAK FINDING & PLOTTING ###
        x_peaks, y_peaks = real_find_peaks(average_unbuttered_fft)
        x_min, y_min = real_find_peaks((average_unbuttered_fft[0], -average_unbuttered_fft[1]))
        #ax.plot(x_peaks, y_peaks, "x")
        #ax.plot(x_min, -y_min, "x")

        ### CURVE PLOTTING ###
        #ax.plot(average_buttered_fft[0], average_buttered_fft[1], label=f'buttered, received, all paths, {f} kHz')
        #unbuttered_fft = our_fft(s.x, fs)
        #plt.plot(unbuttered_fft[0], unbuttered_fft[1], label=f'unbuttered Cycle {s.cycle}, {s.signal_type}, {s.emitter}-{s.receiver}, {s.frequency}')
        envelope_matrix[row_index, :] = average_unbuttered_fft[1]
    print(f'Rendering plots for excitation frequency {f}...')
    
    ### LAYERS ###
    # green for 0-20k, yellow for 20k-40k, orange for 40k-60k, and red for 60k-70?
    # relevant_cycles: Final = ('0', '1', '1000', '10000', '20000', '30000', '40000', '50000', '60000', '70000')
    # Layer 1: 0, 1, 1000
    # Layer 2: 10k, 20k, 30k
    # Layer 3: 40k, 50k
    # Layer 4: 60k, 70k

    ax.fill_between(average_unbuttered_fft[0], envelope_matrix[0, :], envelope_matrix[2, :], color='#f0ff00')
    ax.fill_between(average_unbuttered_fft[0], envelope_matrix[2, :], envelope_matrix[5, :], color='#ffe700')
    ax.fill_between(average_unbuttered_fft[0], envelope_matrix[5, :], envelope_matrix[7, :], color='#ffdb00')
    ax.fill_between(average_unbuttered_fft[0], envelope_matrix[7, :], envelope_matrix[9, :], color='#ffce00')
    
    ### ENVELOPE ###
    upper_envelope = np.max(envelope_matrix, axis=0)
    lower_envelope = np.min(envelope_matrix, axis=0)
    ax.plot(average_unbuttered_fft[0], upper_envelope, color='#030aa7')
    ax.plot(average_unbuttered_fft[0], lower_envelope, color='#f7022a')
    #ax.fill_between(average_buttered_fft[0], lower_envelope, upper_envelope, color='#8cffdb')
    
    ### PLOTTING PARAMETERS ###
    ax.legend()
    ax.set_title(f'FFT: Buttered excitation frequency {f}, received, all paths (averaged), all cycles, residual=False')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Magnitude in dB[-]')
    ax.set_xlim(0, 450000)
    file_path = os.path.join(PLOT_DIR, f'FFT Buttered excitation frequency {f}, received, all paths (averaged), all cycles, residual=False')
    plt.savefig(file_path, dpi = 500)
    plt.show()
    plt.clf()
    plt.close()

'''
for f in allowed_frequencies:
    fig, axs = plt.subplots(8, 1, figsize=(20, 8))
    for j in range(8):
        ax = axs[j]
        envelope_matrix = np.empty([len(relevant_cycles), 12501])
        for row_index, cycle in enumerate(relevant_cycles):
            fc = frequency_collection(cycles=(cycle,), signal_types=('received',), frequency=f, paths=None, residual=False)
            for i, s in enumerate(fc):
                fs = s.sample_frequency
                ### BUTTER ###
                buttered_array = butter_lowpass(s.x, fs, order=20)
                buttered_fft = our_fft(buttered_array, fs, sigma=20)
                ### UNBUTTERED ###
                unbuttered_array = s.x
                unbuttered_fft = our_fft(unbuttered_array, fs, sigma=20)
                
                if i == 0:
                    average_buttered_fft = buttered_fft
                    average_unbuttered_fft = unbuttered_fft
                else:
                    average_buttered_fft += buttered_fft
                    average_unbuttered_fft += unbuttered_fft
            average_buttered_fft /= (i + 1)
            average_unbuttered_fft /= (i + 1)

            ### PEAK FINDING & PLOTTING ###
            x_peaks, y_peaks = real_find_peaks(average_unbuttered_fft)
            x_min, y_min = real_find_peaks((average_unbuttered_fft[0], -average_unbuttered_fft[1]))
            #ax.plot(x_peaks, y_peaks, "x")
            #ax.plot(x_min, -y_min, "x")

            ### CURVE PLOTTING ###
            #ax.plot(average_buttered_fft[0], average_buttered_fft[1], label=f'buttered, received, all paths, {f} kHz')
            #unbuttered_fft = our_fft(s.x, fs)
            #plt.plot(unbuttered_fft[0], unbuttered_fft[1], label=f'unbuttered Cycle {s.cycle}, {s.signal_type}, {s.emitter}-{s.receiver}, {s.frequency}')
            envelope_matrix[row_index, :] = average_unbuttered_fft[1]
        print(f'Rendering plots for excitation frequency {f}...')
        
        ### LAYERS ###
        # green for 0-20k, yellow for 20k-40k, orange for 40k-60k, and red for 60k-70?
        # relevant_cycles: Final = ('0', '1', '1000', '10000', '20000', '30000', '40000', '50000', '60000', '70000')
        # Layer 1: 0, 1, 1000
        # Layer 2: 10k, 20k, 30k
        # Layer 3: 40k, 50k
        # Layer 4: 60k, 70k

        ax.fill_between(average_unbuttered_fft[0], envelope_matrix[0, :], envelope_matrix[2, :], color='#f0ff00')
        ax.fill_between(average_unbuttered_fft[0], envelope_matrix[2, :], envelope_matrix[5, :], color='#ffe700')
        ax.fill_between(average_unbuttered_fft[0], envelope_matrix[5, :], envelope_matrix[7, :], color='#ffdb00')
        ax.fill_between(average_unbuttered_fft[0], envelope_matrix[7, :], envelope_matrix[9, :], color='#ffce00')
        
        ### ENVELOPE ###
        upper_envelope = np.max(envelope_matrix, axis=0)
        lower_envelope = np.min(envelope_matrix, axis=0)
        ax.plot(average_unbuttered_fft[0], upper_envelope, color='#030aa7')
        ax.plot(average_unbuttered_fft[0], lower_envelope, color='#f7022a')
        #ax.fill_between(average_buttered_fft[0], lower_envelope, upper_envelope, color='#8cffdb')
        
        ### PLOTTING PARAMETERS ###
        ax.legend()
        ax.set_title(f'FFT: Buttered excitation frequency {f}, received, all paths (averaged), all cycles, residual=False')
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Magnitude in dB[-]')
        ax.set_xlim(0, 450000)
        file_path = os.path.join(PLOT_DIR, f'FFT Buttered excitation frequency {f}, received, all paths (averaged), all cycles, residual=False')
        plt.savefig(file_path, dpi = 500)
        plt.show()
        plt.clf()
        plt.close()
'''
############################################
### VARYING FREQUENCIES, CONSTANT CYCLES ###
############################################

for cycle in relevant_cycles:
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    # col = 25001
    envelope_matrix = np.empty([len(allowed_frequencies), 12501])
    for row_index, frequency in enumerate(allowed_frequencies):
        sc = signal_collection(cycles=(cycle,), signal_types=('received',), frequencies=(frequency,), emitters=allowed_emitters, receivers=allowed_receivers, residual=False)
        for i, s in enumerate(sc):
            fs = s.sample_frequency
            buttered_array = butter_lowpass(s.x, fs, order=20)
            #buttered_array = s.x
            buttered_fft = our_fft(buttered_array, fs, sigma=20)
            #print(len(buttered_fft), len(s.x))
            if i == 0:
                average_buttered_fft = buttered_fft
            else:
                average_buttered_fft += buttered_fft
        average_buttered_fft /= (i + 1)
        x_peaks, y_peaks = real_find_peaks(average_buttered_fft)
        x_min, y_min = real_find_peaks((average_buttered_fft[0], -average_buttered_fft[1]))
        #ax.plot(x_peaks, y_peaks, "x")
        #ax.plot(x_min, -y_min, "x")
        #ax.plot(average_buttered_fft[0], average_buttered_fft[1], label=f'buttered cycle {cycle}, received, all paths, {frequency} kHz')
        #unbuttered_fft = our_fft(s.x, fs)
        #plt.plot(unbuttered_fft[0], unbuttered_fft[1], label=f'unbuttered Cycle {s.cycle}, {s.signal_type}, {s.emitter}-{s.receiver}, {s.frequency}')
        envelope_matrix[row_index, :] = average_buttered_fft[1] #.flatten()
    ax.fill_between(average_buttered_fft[0], envelope_matrix[0, :], envelope_matrix[1, :], color='#f0ff00')
    ax.fill_between(average_buttered_fft[0], envelope_matrix[1, :], envelope_matrix[2, :], color='#ffe700')
    ax.fill_between(average_buttered_fft[0], envelope_matrix[2, :], envelope_matrix[3, :], color='#ffdb00')
    ax.fill_between(average_buttered_fft[0], envelope_matrix[3, :], envelope_matrix[4, :], color='#ffce00')
    upper_envelope = np.max(envelope_matrix, axis=0)
    print(f'Rendering plots for cycle {cycle}...')
    #print(envelope_matrix)
    #print(upper_envelope)
    lower_envelope = np.min(envelope_matrix, axis=0)
    ax.plot(average_buttered_fft[0], upper_envelope)
    ax.plot(average_buttered_fft[0], lower_envelope)
    #ax.fill_between(average_buttered_fft[0], lower_envelope, upper_envelope, color='#ed0dd9')
    #print(average_buttered_fft.shape)
    ax.legend()
    ax.set_title(f'FFT: Buttered cycle {cycle}, received, all paths (averaged), all allowed_frequencies, residual=False')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Magnitude in dB[-]')
    ax.set_xlim(0, 450000)
    file_path = os.path.join(PLOT_DIR, f'buttered cycle {cycle}, received, all paths (averaged), all allowed_frequencies, residual=False')
    plt.savefig(file_path, dpi = 500)
    plt.show()
    plt.clf()
    plt.close()

'''
for cycle in relevant_cycles:
    fig, axs = plt.subplots(5, 1, figsize=(20, 8))
    # col = 25001
    for j in range(5):
        ax = axs[j]
        envelope_matrix = np.empty([len(allowed_frequencies), 12501])
        for row_index, frequency in enumerate(allowed_frequencies):
            sc = signal_collection(cycles=(cycle,), signal_types=('received',), frequencies=(frequency,), emitters=allowed_emitters, receivers=allowed_receivers, residual=False)
            for i, s in enumerate(sc):
                fs = s.sample_frequency
                buttered_array = butter_lowpass(s.x, fs, order=20)
                #buttered_array = s.x
                buttered_fft = our_fft(buttered_array, fs, sigma=20)
                #print(len(buttered_fft), len(s.x))
                if i == 0:
                    average_buttered_fft = buttered_fft
                else:
                    average_buttered_fft += buttered_fft
            average_buttered_fft /= (i + 1)
            x_peaks, y_peaks = real_find_peaks(average_buttered_fft)
            x_min, y_min = real_find_peaks((average_buttered_fft[0], -average_buttered_fft[1]))
            #ax.plot(x_peaks, y_peaks, "x")
            #ax.plot(x_min, -y_min, "x")
            #ax.plot(average_buttered_fft[0], average_buttered_fft[1], label=f'buttered cycle {cycle}, received, all paths, {frequency} kHz')
            #unbuttered_fft = our_fft(s.x, fs)
            #plt.plot(unbuttered_fft[0], unbuttered_fft[1], label=f'unbuttered Cycle {s.cycle}, {s.signal_type}, {s.emitter}-{s.receiver}, {s.frequency}')
            envelope_matrix[row_index, :] = average_buttered_fft[1] #.flatten()
        ax.fill_between(average_buttered_fft[0], envelope_matrix[0, :], envelope_matrix[1, :], color='#f0ff00')
        ax.fill_between(average_buttered_fft[0], envelope_matrix[1, :], envelope_matrix[2, :], color='#ffe700')
        ax.fill_between(average_buttered_fft[0], envelope_matrix[2, :], envelope_matrix[3, :], color='#ffdb00')
        ax.fill_between(average_buttered_fft[0], envelope_matrix[3, :], envelope_matrix[4, :], color='#ffce00')
        upper_envelope = np.max(envelope_matrix, axis=0)
        print(f'Rendering plots for cycle {cycle}...')
        #print(envelope_matrix)
        #print(upper_envelope)
        lower_envelope = np.min(envelope_matrix, axis=0)
        ax.plot(average_buttered_fft[0], upper_envelope)
        ax.plot(average_buttered_fft[0], lower_envelope)
        #ax.fill_between(average_buttered_fft[0], lower_envelope, upper_envelope, color='#ed0dd9')
        #print(average_buttered_fft.shape)
        # ax.legend()
        ax.set_title(f'FFT: Buttered cycle {cycle}, received, all paths (averaged), all allowed_frequencies, residual=False')
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Magnitude in dB[-]')
        ax.set_xlim(0, 450000)
        ax.set_ylim(-10, 10)
        # file_path = os.path.join(PLOT_DIR, f'buttered cycle {cycle}, received, all paths (averaged), all allowed_frequencies, residual=False')
        # plt.savefig(file_path, dpi = 500)
    fig.tight_layout()
    plt.show()
    plt.clf()
    plt.close()
'''