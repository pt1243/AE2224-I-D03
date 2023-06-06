import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from ..data_io.load_signals import load_data

emitter = [1, 2, 3]
#convert list to numpy array
emitter = np.array(emitter)
receiver = [4, 5, 6]
receiver = np.array(receiver)
frequency = [100, 120, 140, 160, 180]
cycles = ['0', '1', '1000', '10000', '20000', '30000', '40000', '50000', '60000', '70000']

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
                

t = datasetfinal[1,:, 0, 0, 0]
x = datasetfinal[0, :, 0, 0, 0]
plt.plot(t,x)


            
#for now applied on a rondom signal, with added noise, should be added to signal
#t = np.linspace(-1, 1, 201)
#t = t_1
#x_noisy = x_1
#x = (np.sin(2*np.pi*0.75*t*(1-t) + 2.1) +
#0.1*np.sin(2*np.pi*1.25*t + 1) +
# 0.18*np.cos(2*np.pi*3.85*t))
#rng = np.random.default_rng()
#x_noisy = x + rng.standard_normal(len(t)) * 0.08


#second order, lowpass filter, in which b,a are both 1D arrays
b, a = signal.butter(3, 0.05, btype='low') 

#now apply the filter to the noisy signal, with initial conditions
zi = signal.lfilter_zi(b, a)
z, _ = signal.lfilter(b, a, x, zi=zi*x[0])
z2, _ = signal.lfilter(b, a, z, zi=zi*z[0])

#now we apply the filter by using filfilt
y = signal.filtfilt(b, a, x)

plt.figure
plt.plot(t, x ,'b', alpha=0.75)
plt.plot(t, z, 'r--', t, z2, 'r', t, y, 'k')
plt.legend(('noisy signal', 'lfilter, once', 'lfilter, twice',
            'filtfilt'), loc='best')
plt.grid(True)
plt.show()

#def butter_bandpass(lowcut, highcut, t, order):
   # nyq = 0.5 * fs
   # low = lowcut / nyq
   # high = highcut / nyq
  #  b, a = butter(order, [low, high], btype='bandpass', output='ba')
   # return b, a

#def butter_bandpass_filter(data, lowcut, highcut, fs, order):
   # b, a = butter_bandpass(lowcut, highcut, fs, order=order)
   # y = filtfilt(b=b, a=a, x=data)
    # y = lfilter(b=b, a=a, x=data)
    #return y

