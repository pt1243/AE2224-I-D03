import numpy as np
import matplotlib.pyplot as plt
import di_functions as di
from shm_ugw_analysis.data_io.signal import Signal


cycles = ['1', '1000', '10000', '20000', '30000', '40000', '50000', '60000', '70000']
frequencies = list(range(100, 200, 20))
dis = [di.cross_correlation, di.central_spectrum_loss, di.spectrum_loss, di.differential_signal_energy, di.spatial_phase_difference, di.differential_curve_energy]


def plot_di():
    dis = np.array([di.resonance_max_freq(Signal(c, 'received', 2, 5, 180)) for c in cycles])
    dis -= np.min(dis)
    dis /= np.max(dis)
    nb_cycles = list(map(int, cycles))
    plt.plot(nb_cycles, dis)
    plt.show()
    print(di.modified_mann_kendall(dis, nb_cycles))
    pass


def score_dis():
    for d in dis:
        res = np.array([d(cycle=c, emitter=2, receiver=5, frequency=140) for c in cycles])
        print(di.modified_mann_kendall(res, list(map(int, cycles))))
        pass
    pass


plot_di()
