from shm_ugw_analysis.data_io.load import load_data
import numpy as np
import pywt
import matplotlib.pyplot as plt

cycles_list = ['AI', '0', '1', '1000', '10000', '20000', '30000', 
               '40000', '50000', '60000', '70000', 'Healthy']
signal_types_list = ['excitation', 'received']

x_bounds = [0,0.00002] #x domain shown
y_bounds = [-10,10] #y domain shown

file = (3, 0, 2, 4, 140)

def wave(file):
    print('wave called')
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
    return cycle, signal_type, emitter, receiver, frequency, x_bounds, y_bounds

def plot(file):
    print('plot called')
    cycle, signal_type, emitter, receiver, frequency, x_bounds, y_bounds = wave(file)
    y, t, desc = load_data(cycle, signal_type, emitter, receiver, frequency)
    label = str(cycle) + "-" + str(signal_type) + "- em_" + str(emitter) + "- rec_" + str(receiver) + "- freq_" + str(frequency)
    plt.plot(t, y, label=label)
    plt.xlim(x_bounds[0], x_bounds[1])
    plt.ylim(y_bounds[0], y_bounds[1])
    plt.xlabel('t')
    plt.ylabel('y')
    plt.legend()
    plt.savefig('plot.png')

def r_vs_em(file):
    file_list = list(file)
    file_list[1] = 0
    file = tuple(file_list)
    plot(file)
    file_list = list(file)
    file_list[1] = 1
    file = tuple(file_list)
    plot(file)

plot(file)