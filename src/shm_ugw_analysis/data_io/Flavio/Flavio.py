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
    plt.legend()
    plt.savefig('test.png')
    return y,t,desc,frequency

cycle = '0' 
signal_type = 'excitation'
emitter = 1
receiver = 4
frequency = 100

x_bounds = [0,0.00002] #x domain shown
y_bounds = [-10,10] #y domain shown

def r_vs_em(cycle: str, emitter: int, receiver:int, frequency: int, x_bounds: list, y_bounds: list):
    plot(cycle, 'excitation', emitter, receiver, frequency, x_bounds, y_bounds)
    plot(cycle, 'received', emitter, receiver, frequency, x_bounds, y_bounds)

r_vs_em(cycle, emitter, receiver, frequency, x_bounds, y_bounds)

y, t, desc = load_data(cycle, signal_type, emitter, receiver, frequency)

graph = np.array([t,y])
scales = np.arange(y_bounds[0], y_bounds[1]+1)
tstep = desc[0] # time step
Fs = desc[1] # sampling freq
f0 = frequency # signal freq

coef, freqs = pywt.cwt(y, scales, 'gaus1') #Finding CWT using Gaussian wavelet

#Scalogram plot
plt.figure(figsize=(15,10))
plt.imshow(abs(coef), extent=[x_bounds[0], x_bounds[1], y_bounds[1], y_bounds[0]], interpolation='bilinear', cmap='bone', aspect='auto', vmax=abs(coef).max(), vmin=-abs(coef).max())
plt.gca().invert_yaxis()
plt.yticks(np.arange(y_bounds[0],y_bounds[1]+1,1))
plt.xticks(np.arange(x_bounds[0],x_bounds[1]+1,0.00001))
plt.savefig('Scalogram.png')

#Signal plot
plt.figure(figsize=(15, 10))
plt.plot(t,y)
plt.xlim(x_bounds[0],x_bounds[1])
plt.ylim(y_bounds[0],y_bounds[1])
plt.grid(color='gray', linestyle=':', linewidth=0.5)
plt.savefig('signal_plot.png')










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