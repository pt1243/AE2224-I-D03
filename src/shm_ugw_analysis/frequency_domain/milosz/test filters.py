import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal
import scipy as sp
from scipy.signal import savgol_filter
import os
import pandas as pd
import statsmodels.api as sm
from ...data_io.paths import NPZ_DIR


def load_npz(filename):
    """Open the data from a file."""
    contents = np.load(filename)
    x, t, desc = contents['x'], contents['y'], contents['desc']
    return x, t, desc


def load_data(cycle: str, signal_type: str, emitter: int, receiver: int, frequency: int):
    """Loads the data for a given measurement."""

    allowed_cycles = ['AI', '0', '1', '1000', '10000', '20000', '30000', '40000', '50000', '60000', '70000', 'Healthy']
    allowed_signal_types = ['excitation', 'received']
    allowed_emitters = [i for i in range(1, 7)]
    allowed_receivers = allowed_emitters
    allowed_frequencies = [100, 120, 140, 160, 180]

    if cycle not in allowed_cycles:
        raise ValueError(f"Invalid cycle '{cycle}', must be one of {allowed_cycles}")
    if signal_type not in allowed_signal_types:
        raise ValueError(f"Invalid signal type '{signal_type}, must be either 'excitation' or 'received'")
    if emitter not in allowed_emitters:
        raise ValueError(f"Invalid emitter number {emitter}, must be from 1 to 6")
    if receiver not in allowed_receivers:
        raise ValueError(f"Invalid receiver number {receiver}, must be from 1 to 6")
    if emitter in [1, 2, 3]:
        if receiver in [1, 2, 3]:
            raise ValueError(f"Invalid combination of emitter number {emitter} and receiver number {receiver}")
    elif receiver in [4, 5, 6]:
        raise ValueError(f"Invalid combination of emitter number {emitter} and receiver number {receiver}")
    if frequency not in allowed_frequencies:
        raise ValueError(f"Invalid frequency '{frequency}', must be either 100, 120, 140, 160, or 180 [kHz]")
    

    try:
        folder = f'L2_S2_cycle_{int(cycle)}'
    except ValueError:
        folder = f'L2_S2_{cycle}'

    c1_f1 = 'C1' if signal_type == 'excitation' else 'F1'
    freq = str(frequency // 20)
    filename = f'{c1_f1}_mono_1_saine_emetteur_{emitter}_recepteur_{receiver}_excitation_{freq}_00000.npz'

    full_path = NPZ_DIR.joinpath(folder, filename)

    x, t, desc = load_npz(full_path)
    return x, t, desc

data_stacks = 1
t_begin = 0
allfreq = np.zeros((data_stacks, 10))
t_end = 1
samplingfreq = 1000
X=np.linspace(t_begin, t_end, samplingfreq)
Y=np.sin(2*np.pi*50*X) + 0.5*np.sin(2*np.pi*120*X)


mean = np.mean(Y)
std = np.std(Y)
Y_std = (Y-mean)/std
skewness = sp.stats.skew(Y_std)
kurtosis = sp.stats.kurtosis(Y_std)
crest_factor = np.max(Y)/np.sqrt(np.mean(Y**2))
k_factor = np.max(Y)/np.mean(Y)

freq, Psd = signal.welch(Y, len(Y)/(t_begin - t_end) , nperseg=1024)

Y_fft = np.fft.fft(Y)

Mag = np.abs(Y_fft)
Phase = np.angle(Y_fft)



window_size = 11
order = 2   

Y_savgol=savgol_filter(Y, window_size, order)

dir_path = os.path.join(os.getcwd(), 'plots')
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

for i in range(0, data_stacks):
    Y_fft = np.fft.fft(Y)
    frequencies = np.fft.fftfreq(len(Y), (t_begin - t_end)/len(Y))
    indices=np.argsort(Y)
    freqfft=[]
    for k in range(10):
            freqfft.append(frequencies[indices[-k]])
    allfreq[i, :] = freqfft
    np.delete(freqfft, 0)
    plt.plot(frequencies, abs(Y_fft))
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude')
    file_path = os.path.join(dir_path, 'fft'+str(i)+'.png')
    plt.savefig(file_path)
    plt.clf()
    plt.plot(freq, Psd)
    file_path = os.path.join(dir_path, 'psd'+str(i)+'.png')
    plt.savefig(file_path)
    plt.clf()
    plt.plot(X, Y)
    plt.savefig(file_path)
    file_path = os.path.join(dir_path, 'originall'+str(i)+'.png')
    plt.clf()
    plt.plot(X, Y_savgol)
    file_path = os.path.join(dir_path, 'savgol'+str(i)+'.png')
    plt.savefig(file_path)
    plt.clf()
    #save figures in vs code folder python
x_1, t_1, desc_1 = load_data(
    cycle='0',
    signal_type='received',
    emitter=1,
    receiver=4,
    frequency=100
)
print(x_1, t_1, desc_1)         