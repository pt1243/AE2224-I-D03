import numpy as np
import matplotlib.pyplot as plt
from ..data_io.load_signals import Signal , all_paths, relevant_cycles, allowed_frequencies


def cross_correlation(x1, x2):
    return 1 - np.sqrt((np.sum(x1*x2) ** 2) / (np.sum(x1*x1) * np.sum(x2*x2)))


def residual(cycle, actuator, sensor, frequency):
    s = Signal(cycle, 'received', actuator, sensor, frequency)
    x, t = s.x, s.t
    x = x - x.mean()
    s0 = Signal('0', 'received', actuator, sensor, frequency)
    x0 = s0.x
    x0 = x0 - x0.mean()
    xr = x - x0
    return xr, t


def plot_frequencies():
    rel_cycl = relevant_cycles[-6:]
    average = np.zeros((5, 6))

    for f in range(5):
        freq = allowed_frequencies[f]
        for p1, p2 in all_paths:
            results = []
            for c in rel_cycl:
                x, t = residual(c, p1, p2, freq)
                x0, t0 = residual('10000', p1, p2, freq)
                start, end = np.searchsorted(t, 0e-5), np.searchsorted(t, 10e-5)
                results.append(cross_correlation(x[start:end], x0[start:end]))
                pass

            average[f] += np.array(results)
            pass
        pass

    for i in range(5):
        plt.plot([int(u) for u in rel_cycl], average[i])
        pass

    plt.legend(['Frequency 100', 'Frequency 120', 'Frequency 140', 'Frequency 160', 'Frequency 180'])
    plt.xlabel("Cycle number")
    plt.ylabel("Cross-correlation")

    plt.show()
    pass


def plot_emitters(freq=180):
    rel_cycl = relevant_cycles[-6:]
    average = np.zeros((6, 6))

    for p1, p2 in all_paths:
        results = []
        for c in rel_cycl:
            x, t = residual(c, p1, p2, freq)
            x0, t0 = residual('10000', p1, p2, freq)
            start, end = np.searchsorted(t, 0e-5), np.searchsorted(t, 10e-5)
            results.append(cross_correlation(x[start:end], x0[start:end]))
            pass

        average[p1-1] += np.array(results)
        pass

    for i in range(6):
        plt.plot([int(u) for u in rel_cycl], average[i])
        pass

    plt.legend(['Emitter 1', 'Emitter 2', 'Emitter 3', 'Emitter 4', 'Emitter 5', 'Emitter 6'])
    plt.xlabel("Cycle number")
    plt.ylabel("Cross-correlation")

    plt.show()
    pass


# plot_emitters()
plot_frequencies()
