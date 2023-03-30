import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy as sp
from shm_ugw_analysis.data_io.signal import Signal, signal_collection
from scipy.signal import savgol_filter
import os
import pandas as pd
frequency = 100
emitter = [1, 2, 3]
#convert list to numpy array
emitter = np.array(emitter)
receiver = [4, 5, 6]
receiver = np.array(receiver)
frequency = [100, 120, 140, 160, 180]
cycles = ['0', '1', '1000', '10000', '20000', '30000', '40000', '50000', '60000', '70000']
#create a string of ten different colors
#generate list of colors in order lighest to darkest (start from yellow, progress to black)
colors = ['yellow', 'orange', 'pink', 'red', 'purple', 'blue', 'green', 'brown', 'grey', 'black']






#write code loading data into numpy array

degree = 10
dir_path = os.path.join(os.getcwd(), 'plots')
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
fig, (ax1, ax2) = plt.subplots(1, 2)
for h in range(0, len(cycles)):
    Maxfrequency = []
    Maxmag = []
    Maxpsd = []
    Maxfreq = []
    for j in range(0, len(frequency)):
        for z in range(0, len(emitter)):
            for k in range(0 , len(receiver)):
            #create two subplots
                for a in range(0, 2):
                    if a==1:
                        s = Signal(str(cycles[h]), 'excitation', receiver[k], emitter[z], frequency[j])
                    else:
                        s = Signal(str(cycles[h]), 'excitation', emitter[z], receiver[k], frequency[j])
                #plot two subplots
                    
                    t = s.t
                    x = s.x
                    #write the function finding max of the x array
                    


                    samplingfreq = 1/(t[1]-t[0])
                    x_std = (x - np.mean(x))/np.std(x)
                    x_fft = np.fft.fft(x_std)
                    Mag = np.abs(x_fft)
                    frequencies = np.fft.fftfreq(len(x_std), 1/samplingfreq)
                    #apply gaussian filter
                    sort = np.argsort(frequencies)
                    frequencies = frequencies[sort]
                    Mag = Mag[sort]
                    Mag = sp.ndimage.gaussian_filter1d(Mag, sigma=15)
                    def maxr(freq, Mag):
                        max = 0
                        index = 0
                        for i in range(0, len(Mag)):
                            if freq[i] > (frequency[j]-40)*10**3 and freq[i] < (frequency[j]+40)*10**3:
                                if Mag[i] > max:
                                    max = Mag[i]
                                    index = i
                        return max, index

                    #numerical integration of fft using trapezoid rule
                    
                    #compute power spectral density (PSD ) of x_std
                    freq, psd = signal.welch(x_std, samplingfreq, nperseg = 1024, scaling = 'spectrum')
                
                    #sort frequencies and corresponding fft values

                    #integrate fft wrt frequency
                    Mag=Mag/np.trapz(Mag, dx = np.max(frequencies)/len(frequencies))
                    #function evaluating polynomial with coefficients polycfft 

                        

                    #find index of the maximum value
                    #plot the point on the graph
                    
                
                    
                    
                    sort = np.argsort(freq)
                    freq = freq[sort]
                    psd = psd[sort]
                    psd = psd/np.trapz(psd, dx = np.max(freq)/len(freq))
                    max, index = maxr(frequencies, Mag)
                    Maxmag.append(max)
                    Maxfrequency.append(frequencies[index])
                    max, index = maxr(freq, psd)
                    Maxpsd.append(max)
                    Maxfreq.append(freq[index])
    Maxfrequencyaverage = np.mean(Maxfrequency)
    Maxmagaverage = np.mean(Maxmag)
    Maxfreqaverage = np.mean(Maxfreq)
    Maxpsdaverage = np.mean(Maxpsd)
    #plot a labelled point using pyplot.scatter
    ax1.scatter(Maxfrequencyaverage, Maxmagaverage, color = colors[h], label = 'cycles = '+str(cycles[h]))
    ax2.scatter(Maxfreqaverage, Maxpsdaverage, color = colors[h], label = 'cycles = '+str(cycles[h]))

    #sort freq and corresponding PSD values
    #ax1.set_xlim((frequency[j]-20)*10**3, (frequency[j]+20)*10**3)
    #ax1.set_ylim(0.99*np.max(Mag), 1.01*np.max(Mag))
    #write function storing the maximum value of fft and corresponding frequency
    
ax2.grid()
    
    
ax2.set_xlabel('Frequency [Hz]')
ax2.set_ylabel('Power (W)')
    #ax2.set_xlim((frequency[j]-20)*10**3, (frequency[j]+20)*10**3)
    #position legend to the right of the subplots
ax2.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
ax2.set_title('PSD')
ax1.grid()  
ax1.set_xlabel('Frequency [Hz]')
ax1.set_ylabel('Amplitude')
file_path = os.path.join(dir_path, 'FFT+PSD'+'emitter'+str(emitter[z])+'receiver'+str(receiver[k])+ 'frequency'+str(frequency[j])+'KHz'+'.png')
plt.savefig(file_path, dpi = 400)
plt.clf()
    
    