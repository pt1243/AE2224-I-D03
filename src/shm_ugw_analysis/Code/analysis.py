import numpy as np
import matplotlib.pyplot as plt
from shm_ugw_analysis.data_io.load import load_data
import di_fnct as di


cycles = ['0', '1', '1000', '10000', '20000', '30000', '40000', '50000', '60000', '70000']
frequencies = list(range(100, 200, 20))


def data(cycle='0', emitter=1, receiver=4, frequency=100):
    x, t, desc = load_data(cycle=cycle, signal_type='received', emitter=emitter, receiver=receiver, frequency=frequency)
    start, end = np.searchsorted(t, -0.5e-5), np.searchsorted(t, 2.5e-5)
    result = x[start:end]
    result = (result - np.mean(result)/np.std(result))
    return result


def plot_di():
    dis = np.array([di.spatial_phase_difference(cycle=c, emitter=1, receiver=5) for c in cycles])
    dis -= np.min(dis)
    dis /= np.max(dis)
    plt.plot(cycles, dis)
    plt.show()
    pass


plot_di()
