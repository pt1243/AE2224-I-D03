import numpy as np
import matplotlib.pyplot as plt
import di_functions as di


cycles = ['1', '1000', '10000', '20000', '30000', '40000', '50000', '60000', '70000']
frequencies = list(range(100, 200, 20))


def plot_di():
    dis = np.array([di.central_spectrum_loss(cycle=c, emitter=1, receiver=4) for c in cycles])
    dis -= np.min(dis)
    dis /= np.max(dis)
    plt.plot(list(map(int, cycles)), dis)
    plt.show()
    pass


plot_di()
