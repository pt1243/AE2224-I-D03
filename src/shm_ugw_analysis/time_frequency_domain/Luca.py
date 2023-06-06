from ..data_io.load_signals import load_data
import numpy as np
import pywt
import matplotlib.pyplot as plt
from scipy.signal import stft


def wave(file):
    print("wave called")
    global cycles_list
    global signal_types_list
    global x_bounds
    global y_bounds
    cycle_index = file[0]
    signal_type_index = file[1]
    emitter = file[2]
    receiver = file[3]
    frequency = file[4]
    cycle = cycles_list[cycle_index]
    signal_type = signal_types_list[signal_type_index]
    print("wave done")
    return cycle, signal_type, emitter, receiver, frequency, x_bounds, y_bounds


def plot(file):
    print("plot called")
    cycle, signal_type, emitter, receiver, frequency, x_bounds, y_bounds = wave(file)
    y, t, desc = load_data(cycle, signal_type, emitter, receiver, frequency)
    #label = str(cycle) + "-" + str(signal_type) + "- em_" + str(emitter) + "- rec_" + str(receiver) + "- freq_" + str(frequency)
    plt.plot(t, y,) #label=label)
    plt.xlim(x_bounds[0], x_bounds[1])
    plt.ylim(y_bounds[0], y_bounds[1])
    plt.xlabel('t')
    plt.ylabel('y')
    plt.legend()
    plt.title("cycle " + str(cycle) + "  -  em " + str(emitter) + "  -  r " + str(receiver) + "  -  freq " + str(frequency) + " Hz")
    plt.savefig('plot.png')
    print("plot done")


def r_vs_em(file):
    print("r_vs_em called")
    file_list = list(file)
    file_list[1] = 0
    file = tuple(file_list)
    plot(file)
    file_list = list(file)
    file_list[1] = 1
    file = tuple(file_list)
    plot(file)
    print("r_vs_em done")
    

def CWT(file):
    print('CWT called')
    cycle, signal_type, emitter, receiver, frequency, x, y = wave(file)
    signal, t, desc = load_data(cycle, signal_type, emitter, receiver, frequency)
    signal = signal[::1000]
    time = time[::1000]
    freqs = np.arange(0.001, 3, 0.001)
    scales = pywt.scale2frequency('morl', freqs) * len(signal)
    cwtmatr, _ = pywt.cwt(signal, scales, 'morl')
    plt.imshow(np.abs(cwtmatr), extent=[t[0], t[-1] , freqs[-1], freqs[0]],
           aspect='auto', cmap='jet')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    cb = plt.colorbar()
    plt.title(str(cycle) + "  -  " + str(signal_type) + "  -  " + str(emitter) + "  -  " + str(receiver) + "  -  " + str(frequency))
    plt.savefig('CWT.png')
    cb.remove()
    print('CWT done')


def CWT_subplots(file):
    print('CWT_subplots called')
    cycle, signal_type, emitter, receiver, frequency, x, y = wave(file)
    global cycles_list
    plt.suptitle(str(signal_type) + "  -  " + str(emitter) + "  -  " + str(receiver) + "  -  " + str(frequency))
    for i in range(10):
        cycle = cycles_list[i+1]
        signal, t, desc = load_data(cycle, signal_type, emitter, receiver, frequency)
        signal = signal[::1000]
        freqs = np.arange(0.001, 3, 0.001)
        wavelet = morlet
        scales = pywt.scale2frequency(wavelet, freqs) * len(signal)
        cwtmatr, _ = pywt.cwt(signal, scales, wavelet)
        plt.subplot(2, 5, i+1)
        plt.imshow(np.abs(cwtmatr), extent=[t[0], t[-1] , freqs[-1], freqs[0]],
           aspect='auto', cmap='jet')
        #plt.xlabel('Time (s)')
        #plt.ylabel('Frequency (Hz)')
        plt.axis('off')
        plt.title(str(cycles_list[i+1]))# + "  -  " + str(signal_type) + "  -  " + str(emitter) + "  -  " + str(receiver) + "  -  " + str(frequency))
        plt.savefig('CWT_subplots.png')
    print('CWT_subplots done')


def CWT_new(file):
    print('CWT_new called')
    cycle, signal_type, emitter, receiver, frequency, x, y = wave(file)
    signal, t, desc = load_data(cycle, signal_type, emitter, receiver, frequency)
    signal = signal[1:100]
    fs = 1/len(signal) #sampling freq
    # Define wavelet parameters
    wavelet = 'morl'  # Mother wavelet (e.g., Morlet)
    scales = np.arange(0, 6, num=100)  
    
    # Apply CWT to the signal
    coef, freqs = pywt.cwt(signal, scales, wavelet, sampling_period=1/fs)

    plt.figure()
    plt.imshow(np.abs(coef), extent=[t.min(), t.max(), freqs[-1], freqs[0]], cmap='jet', aspect='auto')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar()
    plt.savefig('CWT_new.png')
    print('CWT_new done')


def STFT(file):
    print('STFT called')
    cycle, signal_type, emitter, receiver, frequency, x, y = wave(file)
    signal, t, desc = load_data(cycle, signal_type, emitter, receiver, frequency)
    fs = len(t)/t[-1]
    window = 'hamming'
    f, t, Zxx = stft(signal, fs=1.0, window=window, nperseg=10, noverlap=5, scaling='psd')
    plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=np.max(np.abs(Zxx)), shading='gouraud')
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar()
    plt.savefig('STFT.png')
    print('STFT done')


def morlet_wavelet_imag(frequency: int, standard_deviation: int, t):
    return np.imag(np.exp(2*((-1)**0.5)*np.pi*frequency*t-(t**2)/(2*standard_deviation**2)))
    

def morlet_wavelet_real(frequency: int, standard_deviation: int, t):
    return (np.exp(2*((-1)**0.5)*np.pi*frequency*t-(t**2)/(2*standard_deviation**2)))
  
def morlet(f:float, l:float, dt:float):
    t = np.arange(-l/2, l/2, dt)
    s = l/6
    y = np.exp(2*((-1)**0.5)*np.pi*f*t-(t**2)/(2*s**2))
    y = np.real(y)
    return t, y
def FFT(signal_td):
    return np.fft(signal_td)


def IFFT(signal_fd):
    print('...')


def wavelet_function(x):
    global n
    global f
    s = n/(2*np.pi*f)
    return np.exp(2*((-1)**0.5)*np.pi*f*x-(x**2)/(2*s**2))

def wavelet_definition():
    global n
    global f
    s = n/(2*np.pi*f)
    name = "custom_wavelet,f=" + str(f) + ",n=" + str(n) + ",s=" + str(s)
    custom_scaling_function = wavelet_function
    custom_wavelet = pywt.Wavelet(name, scaling_function=custom_scaling_function)
    return custom_wavelet

def CWT_custom_wavelet(file, custom_wavelet, sampling_frequency, number_of_cycles):
    cycle, signal_type, emitter, receiver, frequency, x_bounds, y_bounds = wave(file)
    title = "N_cycles=" + str(number_of_cycles) + "samp_frequency=" + str(sampling_frequency) + "---" + str(cycle) + "-" + str(signal_type) + "- em_" + str(emitter) + "- rec_" + str(receiver) + "- freq_" + str(frequency)
    y, t, desc = load_data(cycle, signal_type, emitter, receiver, frequency)
    scales = np.arange(1,100)
    coefficients, frequencies = pywt.cwt(y, scales, custom_wavelet, sampling_period=1/sampling_frequency)
    plt.figure()
    plt.imshow(np.abs(coefficients), extent=[t.min(), t.max(), frequencies[-1], frequencies[0]], cmap='jet', aspect='auto')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar()
    plt.title(title)
    plt.savefig('CWT_custom.png')
    


#=================================================
#=================================================
#=================================================

#FFT of signal
#FFT of morlet (with same frequency of excitation frequency maybe)
#point multiplication
#Inverse FFT
#...

cycles_list = ['AI', '0', '1', '1000', '10000', '20000', '30000', 
               '40000', '50000', '60000', '70000']
signal_types_list = ['excitation', 'received']

x_bounds = [0,0.00003] #x domain shown
y_bounds = [-10,10] #y domain shown


file = (1, 1, 2, 6, 140)
#CWT_subplots(file)
#r_vs_em(file)
t = np.arange(-1, 1, 0.001)
f = 10
n = 20
CWT_custom_wavelet(file, wavelet_definition, f, n)
#plt.plot(t, w)
#w_i = morlet_wavelet_imag(f, s, t)
#plt.plot(t, w_i)
#plt.savefig('wavelet')
#l = s*6
#t, y = morlet(f, l, 0.001)
#plt.plot(t, y)
#plt.savefig('wavelet')