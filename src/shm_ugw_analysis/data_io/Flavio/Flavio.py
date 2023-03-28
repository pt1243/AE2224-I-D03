import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from shm_ugw_analysis.data_io.load import load_data
import pywt 
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
    label = "cycle_" + str(cycle) + "-" + str(signal_type) + "-emitter_" + str(emitter) + "-receiver_" + str(receiver) + "-frequency_" + str(frequency)
    plt.plot(t, y, label=label)
    plt.xlim(x_bounds[0],x_bounds[1])
    plt.ylim(y_bounds[0],y_bounds[1])
    plt.xlabel('t')
    plt.ylabel('y')
    plt.legend(fontsize=5)
    plt.savefig('test.png')
    return y,t,desc,frequency

cycle = '0' 
signal_type = 'received'
emitter = 2
receiver = 4
frequency = 100

x_bounds = [0,0.00002] #x domain shown
y_bounds = [-10,10] #y domain shown
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

    scales = np.arange(1, 31)

    coef, freqs = pywt.cwt(y, scales, 'morl') #Finding CWT

    fig = plt.figure(figsize=(15,10))
    plt.imshow(abs(coef), extent=[0, 200, 30, 1], interpolation='bilinear', cmap='hot', aspect='auto', vmax=abs(coef).max(), vmin=-abs(coef).max())
    #plt.gca().invert_yaxis()
    plt.yticks(np.arange(1, 31, 1))
    plt.xticks(np.arange(0, 201, 1))
    label = "cycle_" + str(cycle) + "-" + str(signal_type) + "-emitter_" + str(emitter) + "-receiver_" + str(receiver) + "-frequency_" + str(frequency)
    plt.title(label, fontsize=25)
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.xlim(19,22)
    plt.colorbar()
    plt.savefig('Scalogram.png')
    return abs(coef)

def scalogram_comparison(cycle: str, frequency: int):

    fig, scalogram = plt.subplots(3, 3)

    scalogram[0,0].plot(scalogram_plot(cycle, 'received', 1, 4, frequency))
    scalogram[0,1].plot(scalogram_plot(cycle, 'received', 2, 4, frequency))
    scalogram[0,2].plot(scalogram_plot(cycle, 'received', 3, 4, frequency))
    scalogram[1,0].plot(scalogram_plot(cycle, 'received', 1, 5, frequency))
    scalogram[1,1].plot(scalogram_plot(cycle, 'received', 2, 5, frequency))
    scalogram[1,2].plot(scalogram_plot(cycle, 'received', 3, 5, frequency))
    scalogram[2,0].plot(scalogram_plot(cycle, 'received', 1, 6, frequency))
    scalogram[2,1].plot(scalogram_plot(cycle, 'received', 2, 6, frequency))
    scalogram[2,2].plot(scalogram_plot(cycle, 'received', 3, 6, frequency))

    scalogram[0,0].set_xlim(-19,21)
    scalogram[0,0].set_ylim(1,30)

    plt.tight_layout()
    plt.savefig('Scalogram_comp.png')


#scalogram_plot(cycle, signal_type, emitter, receiver, frequency)
scalogram_comparison(cycle, frequency)


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