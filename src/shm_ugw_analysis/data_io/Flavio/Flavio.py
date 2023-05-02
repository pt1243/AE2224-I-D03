import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from shm_ugw_analysis.data_io.load import load_data
import pywt 
from matplotlib.colors import ListedColormap
#import os

#healthy_path = "C:\\Users\\Gebruiker\\Desktop\\TU DELFT\\Year 2\\Project 2nd semester\\npz_files\\npz_files\\L2_S2_cycle_60000"
#healthy_files = [f for f in os.listdir(healthy_path) if f.endswith('.npz')]
#healthy_lists = [[] for _ in range(len(healthy_files))]

#for i, healthy_file in enumerate(healthy_files):
    #healthy_data = np.load(os.path.join(healthy_path, healthy_file))
    #for array_name in healthy_data.files:
        #healthy_lists[i].append(healthy_data[array_name])

#x = healthy_lists[50][0]
#y = healthy_lists[50][1]
def plot(cycle: str, signal_type: str, emitter: int, receiver: int, frequency: int, x_bounds: list, y_bounds: list):
    y, t, desc = load_data(cycle, signal_type, emitter, receiver, frequency)
    #label = "cycle_" + str(cycle) + "-" + str(signal_type) + "-emitter_" + str(emitter) + "-receiver_" + str(receiver) + "-frequency_" + str(frequency)
    label = "cycle " + str(cycle) + ", " + str(signal_type) + ", " + str(emitter) + "-" + str(receiver) + ", " + str(frequency)
    plt.plot(t, y, label=label)
    plt.xlim(x_bounds[0],x_bounds[1])
    plt.ylim(y_bounds[0],y_bounds[1])
    plt.xlabel('t')
    plt.ylabel('y')
    plt.legend(fontsize=5)
    plt.savefig('test.png')
    return y,t,desc,frequency

cycle = '70000' 
signal_type = 'received'
emitter = 2
receiver = 5
frequency = 180

x_bounds = [0,0.00003] #x domain shown
y_bounds = [-0.1,0.1] #y domain shown
x_bounds1 = [0,0.00002]
y_bounds1 = [-0.25,0.25]


def r_vs_em(cycle: str, emitter: int, receiver:int, frequency: int, x_bounds: list, y_bounds: list):
    plot(cycle, 'excitation', emitter, receiver, frequency, x_bounds, y_bounds)
    plot(cycle, 'received', emitter, receiver, frequency, x_bounds, y_bounds)

def cycle_comparison(emitter: int, receiver:int, frequency: int, x_bounds: list, y_bounds: list):
    plot('0', 'received', emitter, receiver, frequency, x_bounds, y_bounds)
    plot('1', 'received', emitter, receiver, frequency, x_bounds, y_bounds)
    plot('1000', 'received', emitter, receiver, frequency, x_bounds, y_bounds)
    plot('10000', 'received', emitter, receiver, frequency, x_bounds, y_bounds)
    plot('20000', 'received', emitter, receiver, frequency, x_bounds, y_bounds)
    plot('30000', 'received', emitter, receiver, frequency, x_bounds, y_bounds)
    plot('40000', 'received', emitter, receiver, frequency, x_bounds, y_bounds)
    plot('50000', 'received', emitter, receiver, frequency, x_bounds, y_bounds)
    plot('60000', 'received', emitter, receiver, frequency, x_bounds, y_bounds)
    plot('70000', 'received', emitter, receiver, frequency, x_bounds, y_bounds)

def freq_comparison(cycle: str, emitter: int, receiver:int, x_bounds: list, y_bounds: list):
    plot(cycle, 'received', emitter, receiver, 100, x_bounds, y_bounds)
    plot(cycle, 'received', emitter, receiver, 120, x_bounds, y_bounds)
    plot(cycle, 'received', emitter, receiver, 140, x_bounds, y_bounds)
    plot(cycle, 'received', emitter, receiver, 160, x_bounds, y_bounds)
    plot(cycle, 'received', emitter, receiver, 180, x_bounds, y_bounds)


def scalogram_plot(cycle: str, signal_type: str, emitter: int, receiver:int, frequency: int):

    y, t, desc = load_data(cycle, signal_type, emitter, receiver, frequency)

    scales = np.arange(10, 40)

    freqs = pywt.scale2frequency('morl', scales)
    min_freq, max_freq = freqs[0], freqs[-1]
    num_freqs = len(freqs)
    new_freqs = np.linspace(min_freq, max_freq, num_freqs)

    coef, freqs = pywt.cwt(y, scales, 'morl') #Finding CWT


    fig = plt.figure(figsize=(15,10))
    plt.imshow(abs(coef), extent=[0, 0.0002, max_freq, min_freq], interpolation='bilinear', cmap='gist_ncar', aspect='auto', vmax=abs(coef).max(), vmin=-abs(coef).max())
    #plt.gca().invert_yaxis()
    plt.yticks(new_freqs)
    plt.xticks(np.arange(0, 0.0002, 0.000001))
    label = "cycle " + str(cycle) + ", " + str(signal_type) + ", " + str(emitter) + "-" + str(receiver) + ", " + str(frequency)
    plt.title(label, fontsize=25)
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    #plt.xlim(0.190e-5,0.235e-5)
    plt.xlim(1.8e-5,2.3e-5)

    plt.colorbar()
    plt.savefig('Scalogram.png')
    #return fig

def scalogram_subplots(cycle: str, signal_type: str, emitter: int, receiver:int, frequency: int):
    cycles = ['0', '1', '1000', '10000', '20000', '30000', 
               '40000', '50000', '60000', '70000']
    idx = 0
    for i in cycles:
        y, t, desc = load_data(i, signal_type, emitter, receiver, frequency)
        scales = np.arange(10, 20)
        freqs = pywt.scale2frequency('morl', scales)
        min_freq, max_freq = freqs[0], freqs[-1]
        num_freqs = len(freqs)
        new_freqs = np.linspace(min_freq, max_freq, num_freqs)
        coef, freqs = pywt.cwt(y, scales, 'morl') #Finding CWT
        plt.subplot(2, 5, idx + 1)
        plt.imshow(abs(coef), extent=[0, 0.00002, max_freq, min_freq], interpolation='bilinear', cmap='gist_ncar', aspect='auto', vmax=abs(coef).max(), vmin=-abs(coef).max())
        plt.yticks(new_freqs)
        plt.xticks(np.arange(0, 0.00002, 0.000001))
        label = cycles[idx]
        plt.title(label, fontsize=10)
        plt.xlim(0.2e-5,0.215e-5)
        plt.axis('off')
        plt.savefig('Scalogram_subplots.png')
        idx += 1

def scalogram_visual(signal_type: str, emitter: int, receiver:int, frequency: int):
    scalogram_plot('0', signal_type, emitter, receiver, frequency)
    scalogram_plot('1', signal_type, emitter, receiver, frequency)
    scalogram_plot('1000', signal_type, emitter, receiver, frequency)
    scalogram_plot('10000', signal_type, emitter, receiver, frequency)
    scalogram_plot('20000', signal_type, emitter, receiver, frequency)
    scalogram_plot('30000', signal_type, emitter, receiver, frequency)
    scalogram_plot('40000', signal_type, emitter, receiver, frequency)
    scalogram_plot('50000', signal_type, emitter, receiver, frequency)
    scalogram_plot('60000', signal_type, emitter, receiver, frequency)
    scalogram_plot('70000', signal_type, emitter, receiver, frequency)




def wavelet_variance(cycle: str, signal_type: str, emitter: int, receiver:int, frequency: int):

    label = "cycle " + str(cycle) + ", " + str(signal_type) + ", " + str(emitter) + "-" + str(receiver) + ", " + str(frequency)

    y, t, desc = load_data(cycle, signal_type, emitter, receiver, frequency)

    scales = np.arange(1, 31)
    coef, freqs = pywt.cwt(y, scales, 'morl') #Finding CWT

    scalogram = np.square(abs(coef))

    wavelet_variances = np.zeros(scalogram.shape[0])
    for i in range(scalogram.shape[0]): 
        wavelet_variances[i] = np.var(coef[i,:])

    plt.title(label)
    plt.plot(freqs, wavelet_variances, label=label)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Wavelet Variance')
    plt.xlim()
    plt.ylim()
    plt.legend(fontsize=5)
    plt.savefig('WaveletVariance.png')

def wavelet_variance_comp(signal_type: str, emitter: int, receiver:int, frequency: int):

    wavelet_variance('AI', signal_type, emitter, receiver, frequency)
    wavelet_variance('0', signal_type, emitter, receiver, frequency)
    wavelet_variance('1', signal_type, emitter, receiver, frequency)
    wavelet_variance('1000', signal_type, emitter, receiver, frequency)
    wavelet_variance('10000', signal_type, emitter, receiver, frequency)
    wavelet_variance('20000', signal_type, emitter, receiver, frequency)
    wavelet_variance('30000', signal_type, emitter, receiver, frequency)
    wavelet_variance('40000', signal_type, emitter, receiver, frequency)
    wavelet_variance('50000', signal_type, emitter, receiver, frequency)
    wavelet_variance('60000', signal_type, emitter, receiver, frequency)
    wavelet_variance('70000', signal_type, emitter, receiver, frequency)


def PSD(cycle: str, signal_type: str, emitter: int, receiver:int, frequency: int):

    label = "cycle " + str(cycle) + ", " + str(signal_type) + ", " + str(emitter) + "-" + str(receiver) + ", " + str(frequency)

    y, t, desc = load_data(cycle, signal_type, emitter, receiver, frequency)

    coef, freqs = pywt.cwt(y, np.arange(1, 31), 'morl')
    scalogram = np.square(abs(coef))
    psd = np.zeros(scalogram.shape[0])
    for i in range(scalogram.shape[0]):
        freq = pywt.scale2frequency('morl', i+1)
        width = freq/2   # assuming wavelet has bandwidth of 2
        psd[i] = np.sum(scalogram[i,:])/width

    plt.title(label)
    plt.plot(freqs, psd, label=label)
    plt.xlim()
    plt.ylim()
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Power Spectral Density [V^2/Hz]')
    plt.legend(fontsize=5)
    plt.savefig('PSD.png')

def PSD_compare(signal_type: str, emitter: int, receiver:int, frequency: int):
    PSD('AI', signal_type, emitter, receiver, frequency)
    PSD('0', signal_type, emitter, receiver, frequency)
    PSD('10000', signal_type, emitter, receiver, frequency)
    PSD('70000', signal_type, emitter, receiver, frequency)

def Scalogram_New(cycle: str, signal_type: str, emitter: int, receiver:int, frequency: int):
    y, t, desc = load_data(cycle, signal_type, emitter, receiver, frequency)
    
    scales = np.arange(1, 20)
    cwtmatr = signal.cwt(y, signal.morlet2, scales)

    plt.imshow(np.real(cwtmatr), extent=[-1, 1, 1, 31], cmap="gist_ncar", aspect="auto", vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
    #plt.yticks(new_freqs)
    #plt.xticks(np.arange(0, 0.0002, 0.000001))
    label = "cycle " + str(cycle) + ", " + str(signal_type) + ", " + str(emitter) + "-" + str(receiver) + ", " + str(frequency)
    plt.title(label, fontsize=25)
    #plt.xlabel('Time [s]')
    #plt.ylabel('Frequency [Hz]')
    #plt.xlim(0.190e-5,0.235e-5)
    #plt.xlim(1.8e-5,2.3e-5)

    plt.colorbar()
    plt.savefig('Scalogram_new.png')




###################################################################################
#print(pywt.wavelist())
#plot(cycle, signal_type, emitter, receiver, frequency, x_bounds, y_bounds)
#scalogram_plot(cycle, signal_type, emitter, receiver, frequency)
scalogram_subplots(cycle, signal_type, emitter, receiver, frequency)
#Scalogram_New(cycle, signal_type, emitter, receiver, frequency)
scalogram_visual(signal_type, emitter, receiver, frequency)
#scalogram_plot2(cycle, signal_type, emitter, receiver, frequency)
#wavelet_variance(cycle, signal_type, emitter, receiver, frequency)
#PSD(cycle, signal_type, emitter, receiver, frequency)
#PSD_compare(signal_type, emitter, receiver, frequency)
#wavelet_variance_comp(signal_type, emitter, receiver, frequency)
#scalogram_comparison(cycle, frequency)
###################################################################################

#scalogram_plot('0', 'received', 1, 4, 100)
#freq_comparison(cycle, emitter, receiver, x_bounds1, y_bounds1)
#cycle_comparison(emitter, receiver, frequency, x_bounds1, y_bounds1)
#r_vs_em(cycle, emitter, receiver, frequency, x_bounds, y_bounds)

#y, t, desc = load_data(cycle, signal_type, emitter, receiver, frequency)

#graph = np.array([t,y])
#scales = np.arange(1, 31)
#tstep = desc[0] # time step
#Fs = desc[1] # sampling freq
#f0 = frequency # signal freq

#coef, freqs = pywt.cwt(y, scales, 'morl') #Finding CWT


#SCALOGRAM PLOT

#plt.figure(figsize=(15,10))
#plt.imshow(abs(coef), extent=[0, 200, 30, 1], interpolation='bilinear', cmap='hot', aspect='auto', vmax=abs(coef).max(), vmin=-abs(coef).max())
#plt.gca().invert_yaxis()
#plt.yticks(np.arange(1, 31, 1))
#plt.xticks(np.arange(0, 201, 1))
#label = "cycle_" + str(cycle) + "-" + str(signal_type) + "-emitter_" + str(emitter) + "-receiver_" + str(receiver) + "-frequency_" + str(frequency)
#plt.title(label, fontsize=25)
#plt.xlabel('Time')
#plt.ylabel('Frequency')
#plt.xlim(19,22)
#plt.savefig('Scalogram.png')

#Signal plot
#plt.figure(figsize=(15, 10))
#plt.plot(t,y)
#plt.xlim(x_bounds[0],x_bounds[1])
#plt.ylim(y_bounds[0],y_bounds[1])
#plt.grid(color='gray', linestyle=':', linewidth=0.5)
#plt.savefig('signal_plot.png')










#plt.xlim(0,0.00005)
#plt.tlim(-10,10)

#grapht = graph.T
#fs = desc[1]
#F, T, Zxx = signal.stft(t, fs)

#plt.pcolormesh(T, F, np.abs(Zxx), vmin=0, vmax=np.max(np.abs(Zxx)), shading='gouraud')
#plt.title('STFT Magnitude')
#plt.ylabel('Frequency [Hz]')
#plt.xlabel('Time [sec]')
#plt.savefig('STFT.png')