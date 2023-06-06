import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy as sp
from ...data_io.load_signals import Signal, signal_collection
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
#delete indices numpy  array


#write code loading data into numpy array

degree = 10
dir_path = os.path.join(os.getcwd(), 'plots')
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
for h in range(0, len(cycles)):
    counter = 0
    for z in range(0, len(emitter)):
        for k in range(0 , len(receiver)):
            #create two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2)
            for j in range(0, len(frequencies)):
                #plot two subplots
                s = Signal(str(cycles[h]), 'excitation', emitter[z], receiver[k], frequency[j])
                t = s.t
                x = s.x
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
                        if freq[i] > (frequency[j]-20)*10**3 and freq[i] < (frequency[j]+20)*10**3:
                            if Mag[i] > max:
                                max = Mag[i]
                                index = i
                    return max, index
                    counter += 1
                #numerical integration of fft using trapezoid rule
                
                #compute power spectral density (PSD ) of x_std
                freq, psd = signal.welch(x_std, samplingfreq, nperseg = 1024, scaling = 'spectrum')
               
                #sort frequencies and corresponding fft values

                #integrate fft wrt frequency
                Mag=Mag/np.trapz(Mag, dx = np.max(frequencies)/len(frequencies))
                polycfft = np.polyfit(frequencies, Mag, degree)
                #function evaluating polynomial with coefficients polycfft
                def polynomialfft(xp, degree):
                    sum = 0
                    for k in range(degree+1):
                        sum += polycfft[k]*xp**(degree-k)
                    return sum  

                    

                #find index of the maximum value
                #plot the point on the graph
                yfft = polynomialfft(frequencies, degree)
                ax1.plot(frequencies , Mag,   label = 'cycle '+str(cycles[h]))

                ax1.grid()  
                ax1.set_xlabel('Frequency [Hz]')
                ax1.set_ylabel('Amplitude')
                ax1.set_title('FFT'+ 'receiver'+str(receiver[k]) + 'emitter'+str(emitter[z])+'frequency'+str(frequency[j])+'KHz')
                #ax1.set_xlim((frequency[j]-20)*10**3, (frequency[j]+20)*10**3)
                ax1.set_xlim(0.5*10**5, np.max(frequencies))
                #ax1.set_ylim(0.99*np.max(Mag), 1.01*np.max(Mag))
                
                
                sort = np.argsort(freq)
                freq = freq[sort]
                psd = psd[sort]
                psd = psd/np.trapz(psd, dx = np.max(freq)/len(freq))
                #sort freq and corresponding PSD values

                polycpsd = np.polyfit(freq, psd, degree)
                def polynomialpsd(xp, degree):
                    sum = 0
                    for k in range(degree+1):
                        sum += polycpsd[k]*xp**(degree-k)
                    return sum
                ypsd = polynomialpsd(freq, degree)
                ax2.plot(freq, psd, label = 'cycle '+str(cycles[h]))
                ax2.grid()
                ax2.set_xlabel('Frequency [Hz]')
                ax2.set_ylabel('Power (W)')
                #ax2.set_xlim((frequency[j]-20)*10**3, (frequency[j]+20)*10**3)
                ax2.set_xlim(0.5*10**5, np.max(freq))
                ax2.set_ylim(0.95*np.max(psd), 4*np.min(psd))
                ax2.legend()
                ax2.set_title('PSD')
            file_path = os.path.join(dir_path, 'FFT+PSD'+'emitter'+str(emitter[z])+'receiver'+str(receiver[k])+ 'frequency'+str(frequency[j])+'KHz'+'.png')
            plt.savefig(file_path, dpi = 400)
            plt.clf()
