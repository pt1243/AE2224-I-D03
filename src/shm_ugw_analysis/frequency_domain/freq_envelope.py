import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal
import scipy as sp
from scipy.signal import savgol_filter
import os
import pandas as pd
from ..data_io.load_signals import load_data
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

def hl_envelopes_idx(s, dmin=1, dmax=1, split=False):
    """
    Input :
    s: 1d-array, data signal from which to extract high and low envelopes
    dmin, dmax: int, optional, size of chunks, use this if the size of the input signal is too big
    split: bool, optional, if True, split the signal in half along its mean, might help to generate the envelope in some cases
    Output :
    lmin,lmax : high/low envelope idx of input signal s
    """

    # locals min      
    lmin = (np.diff(np.sign(np.diff(s))) > 0).nonzero()[0] + 1 
    # locals max
    lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1 
    
    if split:
        # s_mid is zero if s centered around x-axis or more generally mean of signal
        s_mid = np.mean(s) 
        # pre-sorting of locals min based on relative position with respect to s_mid 
        lmin = lmin[s[lmin]<s_mid]
        # pre-sorting of local max based on relative position with respect to s_mid 
        lmax = lmax[s[lmax]>s_mid]

    # global min of dmin-chunks of locals min 
    lmin = lmin[[i+np.argmin(s[lmin[i:i+dmin]]) for i in range(0,len(lmin),dmin)]]
    # global max of dmax-chunks of locals max 
    lmax = lmax[[i+np.argmax(s[lmax[i:i+dmax]]) for i in range(0,len(lmax),dmax)]]
    
    return lmin,lmax

# plot
#lmin, lmax = hl_envelopes_idx(s)
#plt.plot(t,s,label='signal')
#plt.plot(t[lmin], s[lmin], 'r', label='low')
#plt.plot(t[lmax], s[lmax], 'g', label='high')

for h in range(0, len(frequency)):
    z=0
    k=0
    plt.subplots(1, 1, figsize=(10, 15))
    j=0
    #plot two subplots
    x = datasetfinal[2*j, :, k, z, h]
    t = datasetfinal[2*j+1, :, k, z, h]
    x_std = (x - np.mean(x))/np.std(x)
    x_fft = np.fft.fft(x_std)
    #apply gaussian filter
    x_fft = sp.ndimage.gaussian_filter1d(x_fft, sigma=10)
    #numerical integration of fft using trapezoid rule
    Mag = np.abs(x_fft)
    Mag = Mag/np.trapz(Mag, dx = 1/samplingfreq)
    frequencies = np.fft.fftfreq(len(x_std), 1/samplingfreq)
    # enveloping
    lmin, lmax = hl_envelopes_idx(Mag, dmin=1, dmax=1)
    # FREQUENCY-DOMAIN PLOTS (FREQ V AMP)
    plt.plot(frequencies, Mag, label = 'cycle '+str(cycles[j]))
    plt.plot(frequencies[lmin], Mag[lmin], 'r', label='low')
    plt.plot(frequencies[lmax], Mag[lmax], 'b', label='high')
    plt.grid()
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude')
    plt.xlim(0.5*10**5, 7*10**5)
    plt.ylim(0, 2000)
    plt.legend()
    plt.title('FFT'+ 'receiver'+str(receiver[k]) + 'emitter'+str(emitter[z])+'frequency'+str(frequency[h])+'KHz')
    freq, Psd = signal.welch(x_std, samplingfreq, nperseg=1024)
    PSD = Psd/np.trapz(Psd, dx = 1/samplingfreq)
    file_path = os.path.join(dir_path, 'FFT'+'emitter'+str(emitter[z])+'receiver'+str(receiver[k])+ 'frequency'+str(frequency[h])+'KHz'+'.png')
    plt.savefig(file_path, dpi=500)
    plt.clf()
    #set plot size
    # POWER SPECTRAL DENSITY PLOTS (FREQ V PSD)
    lmin, lmax = hl_envelopes_idx(Psd)
    #plt.plot(freq, Psd, label = 'cycle '+str(cycles[j]))
    plt.plot(freq[lmin], Psd[lmin], 'r', label='low')
    plt.plot(freq[lmax], Psd[lmax], 'b', label='high')
    plt.grid()
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Power (W)')
    plt.xlim(0.5*10**5, 7*10**5)
    plt.legend()
    plt.title('PSD'+ 'receiver'+str(receiver[k]) + 'emitter'+str(emitter[z])+'frequency'+str(frequency[h])+'KHz')
    file_path = os.path.join(dir_path, 'PSD'+'emitter'+str(emitter[z])+'receiver'+str(receiver[k])+ 'frequency'+str(frequency[h])+'KHz'+'.png')
    plt.savefig(file_path, dpi=500)
    plt.clf()
#file_path = os.path.join(dir_path, 'FFT+PSD'+'emitter'+str(emitter[z])+'receiver'+str(receiver[k])+ 'frequency'+str(frequency[h])+'KHz'+'.png')
#plt.savefig(file_path, dpi = 500)
#plt.clf()