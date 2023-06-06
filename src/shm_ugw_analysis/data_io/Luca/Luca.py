from shm_ugw_analysis.data_io.load import load_data
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
    signal = signal[::200]
    t = t[::200]
    freqs = np.arange(0.001, 1.2, 0.001)
    scales = pywt.scale2frequency('morl', freqs) * len(signal)
    cwtmatr, _ = pywt.cwt(signal, scales, 'morl')
    plt.imshow(np.abs(cwtmatr), extent=[t[0], t[-1] , freqs[-1], freqs[0]],
           aspect='auto', cmap='jet')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.colorbar()
    #plt.title(str(cycle) + "  -  " + str(signal_type) + "  -  " + str(emitter) + "  -  " + str(receiver) + "  -  " + str(frequency))
    #plt.title(str(emitter) + "\u2192" + str(receiver) + "  (" + str(frequency) + " kHz) - cycle " + str(cycle))
    plt.savefig('Final CWT_'+ str(emitter)+ '_' + str(receiver)+ '_' +str(frequency) +'_'+str(cycle)+ '.png')
    print('CWT done')
#file = (cycle index, type_index, em, r, freq)
#file = (7, 1, 6, 2, 180)


# 34   15   62
# 120  160  180
def CWT_subplots(file):
    print('CWT_subplots called')
    cycle, signal_type, emitter, receiver, frequency, x, y = wave(file)
    global cycles_list
    plt.suptitle(str(signal_type) + "  -  " + str(emitter) + "  -  " + str(receiver) + "  -  " + str(frequency))
    sums = []
    cycles = []
    for i in range(10):
        cycle = cycles_list[i+1]
        signal, t, desc = load_data(cycle, signal_type, emitter, receiver, frequency)
        signal = signal[::200]
        t = t[::200]
        freqs = np.arange(0.001, 1, 0.001)
        scales = pywt.scale2frequency('morl', freqs) * len(signal)
        #scales = np.arange(10, 40)
        cwtmatr, _ = pywt.cwt(signal, scales, 'morl')
        plt.subplot(3, 4, i+1)
        plt.imshow(np.abs(cwtmatr), extent=[t[0], t[-1] , freqs[-1], freqs[0]],
           aspect='auto', cmap='jet')
        #plt.xlabel('Time (s)')
        #plt.ylabel('Frequency (Hz)')
        plt.axis('off')
        plt.title(str(cycles_list[i+1]))# + "  -  " + str(signal_type) + "  -  " + str(emitter) + "  -  " + str(receiver) + "  -  " + str(frequency))
        plt.savefig('CWT_subplots.png')
        sums.append(np.average(cwtmatr))
        cycles.append(i)

    

    plt.plot(cycles, sums)
    plt.savefig('CWT_subplots.png')
    print('CWT_subplots done')

def average_col_f(file):
    #plt.clf()
    cycle, signal_type, emitter, receiver, frequency, x, y = wave(file)
    global cycles_list
    title = str(emitter)# + " - " + str(receiver)
    add = np.zeros(9)
    #sums = np.zeros(9)
    for f in range(0,5):
        
        sums = []
        cycles = []
        myxticks = []
        f_list = [100, 120, 140, 160, 180]
        frequency = f_list[f]
        for i in range(1,10):
            cycle = cycles_list[i+1]
            signal, t, desc = load_data(cycle, signal_type, emitter, receiver, frequency)
            signal = signal[::200]
            t = t[::200]
            freqs = np.arange(0.001, 1.2, 0.01)
            scales = pywt.scale2frequency('morl', freqs) * len(signal)
            cwtmatr, _ = pywt.cwt(signal, scales, 'morl')
            avg = np.average(cwtmatr)
            #sums[i-1] += avg
            sums.append(avg)
            cycles.append(i)
            myxticks.append(cycle)
            add[i-1] += avg
            
    #sums_norm = np.abs(sums) / np.linalg.norm(sums)
    sums_norm = sums / sums[0]
    #for i in range(len(sums_norm)):
        #if sums_norm[i] < 0.27:
            #failure_cycle[receiver][emitter] = int(myxticks[i])
            #break
        #failure_cycle[receiver][emitter] = "NF"
    #if receiver != 3 and receiver != 6:
        #failure.append(sums_norm[-1])
    #label = str(frequency) + ' kHz'
    label = "average over all excitation frequencies"
    #plt.clf()
    colors = ['dodgerblue', 'mediumblue', 'darkorchid', 'crimson', 'coral']

    plt.xticks(cycles, myxticks)
    plt.plot(cycles, sums_norm, color='k')
    #plt.legend()
    plt.xlabel('Cycle Number [-]')
    plt.ylabel('Coefficient [-]')
    #plt.title(str(emitter) + "\u2192" + str(receiver))
    plt.savefig("3paths Final 1 average " + str(title))
    #plt.plot(cycles, add/5, label = "average", linewidth=3, color='k')
    #plt.legend()
    #plt.savefig("average cols  " + str(title))



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
    plt.savefig('CWT_new_3_4_120.png')
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


# 34   15   62
# 120  140  160

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

#file = (cycle index, type_index, em, r, freq)
file = (10, 1, 6, 2, 160)
#CWT(file)
b = True
if b:
    for c in [1]:
        plt.clf()
        file = (c, 1, 3, 4, 120)
        average_col_f(file)
        file = (c, 1, 3, 5, 120)
        average_col_f(file)
        file = (c, 1, 3, 6, 120)
        average_col_f(file)
        plt.clf()
        file = (c, 1, 1, 4, 140)
        average_col_f(file)
        file = (c, 1, 1, 5, 140)
        average_col_f(file)
        file = (c, 1, 1, 6, 140)
        average_col_f(file)
        plt.clf()
        file = (c, 1, 6, 1, 160)
        average_col_f(file)
        file = (c, 1, 6, 2, 160)
        average_col_f(file)
        file = (c, 1, 6, 3, 160)
        average_col_f(file)
#CWT_subplots(file)
#average_col_f(file)
#failure = []
failure_cycle = [
    ["", "1", "2", "3", "4", "5", "6"],
    ["1", "", "", "", 0, 0, 0],
    ["2", "", "", "", 0, 0, 0],
    ["3", "", "", "", 0, 0, 0],
    ["4", 0, 0, 0, "", "", ""],
    ["5", 0, 0, 0, "", "", ""],
    ["6", 0, 0, 0, "", "", ""],
]

failure_c = [
    ['silver', 'silver', 'silver', 'silver', 'silver', 'silver', 'silver'],
    ['silver', 'grey', 'grey', 'grey', 'w', 'w', 'w'],
    ['silver', 'grey', 'grey', 'grey', 'w', 'w', 'w'],
    ['silver', 'grey', 'grey', 'grey', 'w', 'w', 'w'],
    ['silver', 'w', 'w', 'w', 'grey', 'grey', 'grey'],
    ['silver', 'w', 'w', 'w', 'grey', 'grey', 'grey'],
    ['silver', 'w', 'w', 'w', 'grey', 'grey', 'grey'],
]

a = False
if a:
    for i in [1, 2, 3]:
        for j in [4, 5, 6]:
            file = (1, 1, i, j, 100)
            print(i, j)
            average_col_f(file)


            plt.clf()
            failure_cycle_mat = failure_cycle
            plt.axis('off')
            t = plt.table(failure_cycle_mat, loc = 'center', cellColours = failure_c, cellLoc = 'center')
            t.scale(1,3)
            plt.savefig("predicted failures")

            file = (1, 1, j, i, 100)
            print(j, i)
            average_col_f(file)

            plt.clf()
            failure_cycle_mat = failure_cycle


            plt.axis('off')
            t = plt.table(failure_cycle_mat, loc = 'center', cellColours = failure_c, cellLoc = 'center')
            t.scale(1,3)
            plt.savefig("predicted failures")
#print(np.mean(failure))
#0.20202258847669688
cycles = [1, 1000, 10000, 20000, 30000, 40000, 50000, 60000, 70000]
count = np.zeros(len(cycles))
c_list = []
e = False
if e:
    for c in range(len(cycles)):
        for i in [1, 2, 3]:
            for j in [4, 5, 6]:
                if failure_cycle_mat[i][j] != "NF":
                    if failure_cycle_mat[i][j] <= cycles[c]:
                        count[c] += 1
                if failure_cycle_mat[j][i] != "NF":
                    if failure_cycle_mat[j][i] <= cycles[c]:
                        count[c] += 1
        c_list.append(c)
    plt.clf()
    myxticks = cycles
    myxticks.append('n')
    c_list.append(c_list[-1]+1)
    count_2 = [0]
    for i in count:
        count_2.append(i)
    plt.xticks(c_list, myxticks)
    plt.yticks(range(0,19))
    plt.step(c_list, count_2, linewidth = 2, color='k')
    plt.xlim(0,8.05)
    plt.ylim(-0.1, 18.1)
    plt.xlabel("Cycle Number [-]")
    plt.ylabel("Number of Active Paths [-]")
    #plt.title("Paths activation")
    plt.grid(axis='y')
    plt.savefig("count_cycles")
#r_vs_em(file)

x = [0,1,2,3,4]
y = [0,0,1,2,5]
x_lab = ['1', '1000', '10000', '20000', 'n']
plt.xticks(x, x_lab)
plt.yticks(range(0,11))
plt.step(x, y, linewidth=2, color='green')
plt.xlim(0,3.1)
plt.ylim(-0.2, 10.2)
plt.grid(axis = 'y')
plt.savefig("test.png")




f = 10
n = 20

#plt.plot(t, w)
#w_i = morlet_wavelet_imag(f, s, t)
#plt.plot(t, w_i
#plt.savefig('wavelet')
#l = s*6
#t, y = morlet(f, l, 0.001)
#plt.plot(t, y)
#plt.savefig('wavelet')

class CustomWavelet(pywt.Wavelet):
    def __init__(self, name):
        self.name = name
        def wavelet_function(x):
            global n
            global f
            s = n/(2*np.pi*f)
            # define your custom wavelet function here
            return np.exp(2*((-1)**0.5)*np.pi*f*x-(x**2)/(2*s**2))
        
        filter_bank = [1, wavelet_function, 
                    pywt.upcoef('d', wavelet_function, 2), 
                    pywt.upcoef('d', wavelet_function, 2)[::-1]]

        super().__init__(name, filter_bank)

def custom_wavelet(name):
    
    return CustomWavelet(name)

def CWT_custom_wavelet(file, custom_wavelet_obj, sampling_frequency, number_of_cycles):
    cycle, signal_type, emitter, receiver, frequency, x_bounds, y_bounds = wave(file)
    title = "N_cycles=" + str(number_of_cycles) + "samp_frequency=" + str(sampling_frequency) + "---" + str(cycle) + "-" + str(signal_type) + "- em_" + str(emitter) + "- rec_" + str(receiver) + "- freq_" + str(frequency)
    y, t, desc = load_data(cycle, signal_type, emitter, receiver, frequency)
    scales = np.arange(1,100)
    coefficients, frequencies = pywt.cwt(y, scales, custom_wavelet_obj)
    plt.figure()
    plt.imshow(np.abs(coefficients), extent=[t.min(), t.max(), frequencies[-1], frequencies[0]], cmap='jet', aspect='auto')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar()
    plt.title(title)
    plt.savefig('CWT_custom.png')
    print('done_custom')

#custom_wavelet_obj = custom_wavelet('my_custom_wavelet')
#CWT_custom_wavelet(file, custom_wavelet_obj, f, n)

