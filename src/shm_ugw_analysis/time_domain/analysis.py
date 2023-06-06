import numpy as np
import matplotlib.pyplot as plt
from .di_functions import (
    cross_correlation,
    central_spectrum_loss,
    spectrum_loss,
    differential_signal_energy,
    spatial_phase_difference,
    differential_curve_energy,
    modified_mann_kendall,
    resonance_max_val,
    resonance_max_freq
)
from ..data_io.load_signals import Signal

print('successful import')

cycles = ['1', '1000', '10000', '20000', '30000', '40000', '50000', '60000', '70000']
nb_cycles = list(map(int, cycles))
frequencies = list(range(100, 200, 20))
dis = [cross_correlation, central_spectrum_loss, spectrum_loss, differential_signal_energy, spatial_phase_difference, differential_curve_energy]


def plot_di():
    dis = np.array([resonance_max_val(Signal(c, 'received', 6, 3, 180)) for c in cycles])
    dis -= np.min(dis)
    dis /= np.max(dis) if np.max(dis) > 0 else 1
    plt.plot(nb_cycles, dis)
    plt.show()
    print(modified_mann_kendall(dis, nb_cycles))
    pass


def score_dis():
    for d in dis:
        res = np.array([d(cycle=c, emitter=2, receiver=5, frequency=140) for c in cycles])
        print(modified_mann_kendall(res, list(map(int, cycles))))
        pass
    pass


def compare_mon():
    mon = np.zeros((5, 18))
    for i in range(5):
        f = frequencies[i]
        for j in range(1, 4):
            for k in range(4, 7):
                mon[i, (j-1)*3+k-4] = modified_mann_kendall(np.array([resonance_max_val(Signal(c, 'received', j, k, f)) for c in cycles]), nb_cycles)
                mon[i, (j-1)*3+k-4] = modified_mann_kendall(np.array([resonance_max_val(Signal(c, 'received', j, k, f)) for c in cycles]), nb_cycles)
                pass
            pass
        pass
    # plt.plot(frequencies, mon)
    plt.show()
    pass


def compare_var():
    mon = []
    var = []
    for f in frequencies:
        d = np.array([resonance_max_val(Signal(c, 'received', 3, 4, f)) for c in cycles])
        mon.append(modified_mann_kendall(d, nb_cycles))
        var.append(np.std(d))
        pass
    var = 25*np.array(var)
    plt.plot(frequencies, mon, 'r')
    plt.plot(frequencies, var, 'b')
    plt.show()
    pass


plot_di()
