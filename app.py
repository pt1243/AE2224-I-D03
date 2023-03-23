import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal
import scipy as sp
from scipy.signal import savgol_filter
import os
import pandas as pd
from data_io.load import load_data
frequency = 100
emitter = [1, 2, 3]
#convert list to numpy array
emitter = np.array(emitter)
receiver = [4, 5, 6]
receiver = np.array(receiver)
frequency = [100, 120, 140, 160, 180]
cycles = ['0', '1', '1000', '10000', '20000', '30000', '40000', '50000', '60000', '70000']
#delete indices numpy  array

datasetfinal = np.zeros((2*len(cycles), 25001, len(receiver), len(emitter), len(frequency)))
for j in range(0, len(frequency)):
    for z in range(0, len(emitter)):
        for k in range(0, len(receiver)):

            for i in range(0, len(cycles)):  
                datasetfinal[2*i+1, :, k, z, j], datasetfinal[2*i, :, k, z, j], desc = load_data(
            
                cycle=cycles[i],
                signal_type='received',
                emitter=emitter[z],
                receiver=receiver[k],
                frequency=frequency[j])

#write code loading data into numpy array
t = datasetfinal[1,:, 0, 0, 0]
x = datasetfinal[0, :, 0, 0, 0]

t_begin = t[0]
t_end = t[-1]
deltat = t[1]-t[0]
samplingfreq = 1/(t[1]-t[0])
plt.plot(t, x)


mean = np.mean(x)
std = np.std(x)
x_std = (x-mean)/std
skewness = sp.stats.skew(x_std)
kurtosis = sp.stats.kurtosis(x_std)
crest_factor = np.max(x)/np.sqrt(np.mean(x**2))
k_factor = np.max(x)/np.mean(x)



freq, Psd = signal.welch(x_std, samplingfreq, nperseg=1024)

x_fft = np.fft.fft(x_std)

Mag = np.abs(x_fft)
Phase = np.angle(x_fft)

frequencies = np.fft.fftfreq(len(x_std), 1/samplingfreq)


#delete all zero values in numpy array
window_size = 3
order = 1

x_savgol=savgol_filter(x_std, window_size, order)
i=10

dir_path = os.path.join(os.getcwd(), 'plots')
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
for h in range(0, len(frequency)):
    for z in range(0, len(emitter)):
        for k in range(0 , len(receiver)):

            for j in range(0, len(cycles)):
                x = datasetfinal[2*j, :, k, z, h]
                t = datasetfinal[2*j+1, :, k, z, h]
                x_std = (x - np.mean(x))/np.std(x)
                x_fft = np.fft.fft(x_std)

                #apply gaussian filter
                x_fft = sp.ndimage.gaussian_filter1d(x_fft, sigma=10)
                #numerical integration of fft
                for k in range(0,  len(t)-1):
                    int += (x_fft[k]+x_fft[k+1])/2*(t[k+1]-t[k])
                Mag = np.abs(x_fft)
                frequencies = np.fft.fftfreq(len(x_std), 1/samplingfreq)
                plt.plot(frequencies, Mag/int, label = 'cycle '+str(cycles[j]))
                plt.grid()
                plt.xlabel('Frequency [Hz]')
                plt.ylabel('Amplitude')
                plt.xlim(0, 7*10**5)
                plt.legend()
                plt.title('FFT'+ 'receiver'+str(receiver[k]) + 'emitter'+str(emitter[z])+'frequency'+str(frequency[h])+'KHz')
                file_path = os.path.join(dir_path, 'FFT'+ 'emitter'+str(emitter[z])+'receiver'+str(receiver[k])+'frequency'+str(frequency[h])+'KHz'+'.png') 
                freq, Psd = signal.welch(x_std, samplingfreq, nperseg=1024)
                #set plot size
                plt.plot(freq, Psd, label = 'cycle '+str(cycles[i]))
                plt.grid()
                plt.xlabel('Frequency [Hz]')
                plt.ylabel('Power (W)')
                plt.xlim(0, 7*10**5)
                plt.legend()
                plt.title('PSD'+ 'receiver'+str(receiver[k]) + 'emitter'+str(emitter[z])+'frequency'+str(frequency[h])+'KHz')
                file_path = os.path.join(dir_path, 'PSD'+'emitter'+str(emitter[z])+'receiver'+str(receiver[k])+ 'frequency'+str(frequency[h])+'KHz'+'.png')
                plt.savefig(file_path, dpi = 400)
           



#plot two subplots
    