import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal
from scipy.signal import savgol_filter, welch
import os
import pandas as pd
from shm_ugw_analysis.data_io.load import load_data, allowed_emitters, allowed_receivers, allowed_frequencies
from shm_ugw_analysis.data_io.signal import Signal, signal_collection
from shm_ugw_analysis.data_io.paths import ROOT_DIR
import pathlib


PLOT_DIR = ROOT_DIR.joinpath('plots')
if not pathlib.Path.exists(PLOT_DIR):
    pathlib.Path.mkdir(PLOT_DIR)

def fft_and_psd_plots(sc: signal_collection, bin_width, emitter, receiver):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    for s in sc:
        s: Signal
        x = s.x
        t = s.t
        # Parameters
        fs = s.sample_frequency
        nperseg = int(fs/bin_width)
        # Normalizing
        x_std = (x - np.mean(x))/np.std(x)

        # Standard FFT Method
        #x_fft = endaq.calc.fft.aggregate_fft(x_std)
        
        # Welch FFT Method
        x_fft_welch, y_fft_welch = welch(x, fs=fs, nperseg=nperseg)
        #x_fft = np.fft.fft(x_std)

        # Gaussian Filter
        #x_fft = sp.ndimage.gaussian_filter1d(x_fft_welch, sigma=10)
        
        # numerical integration of fft using trapezoid rule
        Mag = np.abs(y_fft_welch)
        frequencies = x_fft_welch
        #frequencies = np.fft.fftfreq(len(x_std), 1/samplingfreq)
        #integrate fft wrt frequency
        # Mag=Mag/np.trapz(Mag, dx = np.max(frequencies)/len(frequencies))
        ax1.plot(frequencies, Mag, label = f'cycle {s.cycle}')

        freq, Psd = signal.welch(x_std, fs, nperseg=nperseg)
        ax2.plot(freq, Psd, label = f'cycle {s.cycle}')

    ax1.grid()  
    ax1.set_xlabel('Frequency [Hz]')
    ax1.set_ylabel('Amplitude')
    ax1.set_xlim(0.5*10**5, 7*10**5)
    ax1.legend()
    # ax1.set_title('FFT'+ 'receiver'+str(receiver[k]) + 'emitter'+str(emitter[z])+'frequency'+str(frequency[h])+'KHz')
    ax1.set_title(f'FFT emitter {emitter} receiver {receiver} frequency {s.frequency} kHz')
    
    # PSD = Psd/np.trapz(Psd, dx = freq[-1]/len(freq))
    #set plot size
    
    ax2.grid()
    ax2.set_xlabel('Frequency [Hz]')
    ax2.set_ylabel('Power (W)')
    ax2.set_xlim(0.5*10**5, 7*10**5)
    ax2.legend()
    # ax2.set_title('PSD'+ 'receiver'+str(receiver[k]) + 'emitter'+str(emitter[z])+'frequency'+str(frequency[h])+'KHz')
    ax2.set_title(f'PSD emitter {emitter} receiver {receiver} frequency {s.frequency} kHz')
    # file_path = os.path.join(dir_path, 'FFT+PSD'+'emitter'+str(emitter[z])+'receiver'+str(receiver[k])+ 'frequency'+str(frequency[h])+'KHz'+'.png')
    file_path = os.path.join(PLOT_DIR, f'FFT+PSD_emitter_{emitter}_receiver_{receiver}_frequency_{s.frequency}_kHz.png')
    plt.savefig(file_path, dpi = 500)
    return


cycles=['0', '1', '10000', '20000', '30000', '40000', '50000', '60000', '70000']
# cycles = ['0']
signal_types=['excitation']
# emitters=list(allowed_emitters)
emitters = [1]
# receivers=list(allowed_receivers)
receivers = [4]
frequencies=list(allowed_frequencies)
# frequencies = [100]

for frequency in frequencies:
    for emitter in emitters:
        for receiver in receivers:
            try:
                sc = signal_collection(cycles=tuple(cycles), signal_types=tuple(signal_types), emitters=[emitter], receivers=[receiver], frequencies=[frequency])
                fft_and_psd_plots(sc, 1000, emitter, receiver)
            except Exception:
                continue
        


# for s in sc:
#     fft_and_psd_plots(s, 5000)

# fft_and_psd_plots(Signal('1000', 'received', 1, 4, 100), 5000)