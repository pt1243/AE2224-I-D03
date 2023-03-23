import numpy as np
import matplotlib.pyplot as plt
import di_functions as di


cycles = ['1', '1000', '10000', '20000', '30000', '40000', '50000', '60000', '70000']
frequencies = list(range(100, 200, 20))


def plot_di():
    dis = np.array([di.cross_correlation(cycle=c, emitter=2, receiver=5, frequency=140) for c in cycles])
    dis -= np.min(dis)
    dis /= np.max(dis)
    nb_cycles = list(map(int, cycles))
    plt.plot(nb_cycles, dis)
    plt.show()
    print(di.modified_mann_kendall(nb_cycles, nb_cycles))
    pass


# plot_di()

print(di.modified_mann_kendall([-1, 2, 3, -4, 5], [1, 2, 3, 4, 5]))
