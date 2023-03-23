import numpy as np
from shm_ugw_analysis.data_io.signal import Signal


def data(cycle='0', emitter=1, receiver=4, frequency=100):
    s = Signal(cycle, 'received', emitter, receiver, frequency)
    x, t = s.x, s.t

    start, end = np.searchsorted(t, -0.5e-5), np.searchsorted(t, 2.5e-5)

    result = x[start:end]
    result = (result - np.mean(result)/np.std(result))
    return result


def cross_correlation(cycle='70000', emitter=1, receiver=4, frequency=100):
    x0 = data(cycle='0', emitter=emitter, receiver=receiver, frequency=frequency)
    x = data(cycle=cycle, emitter=emitter, receiver=receiver, frequency=frequency)
    return 1 - np.sqrt((np.sum(x0*x) ** 2) / (np.sum(x0*x0) * np.sum(x*x)))
    pass


def spatial_phase_difference(cycle='70000', emitter=1, receiver=4, frequency=100):
    x0 = data(cycle='0', emitter=emitter, receiver=receiver, frequency=frequency)
    x = data(cycle=cycle, emitter=emitter, receiver=receiver, frequency=frequency)
    d = x / np.sqrt(np.sum(x*x))
    a = np.sum(d*x0) / np.sum(x0*x0)
    return np.sum((d-a*x0)**2)


def spectrum_loss(cycle='70000', emitter=1, receiver=4, frequency=100):
    x0 = np.fft.fft(data(cycle='0', emitter=emitter, receiver=receiver, frequency=frequency))
    x = np.fft.fft(data(cycle=cycle, emitter=emitter, receiver=receiver, frequency=frequency))
    return np.sum(np.abs(x0-x)) / np.sum(np.abs(x0))


def central_spectrum_loss(cycle='70000', emitter=1, receiver=4, frequency=100):
    x0 = np.fft.fft(data(cycle='0', emitter=emitter, receiver=receiver, frequency=frequency))
    x = np.fft.fft(data(cycle=cycle, emitter=emitter, receiver=receiver, frequency=frequency))
    a, b = np.max(x0), np.max(x)
    return (a-b)/a


def differential_curve_energy(cycle='70000', emitter=1, receiver=4, frequency=100):
    x0 = data(cycle='0', emitter=emitter, receiver=receiver, frequency=frequency)
    x = data(cycle=cycle, emitter=emitter, receiver=receiver, frequency=frequency)
    b = x0 - x
    return np.sum((b[1:] - b[:-1])**2) / np.sum((x0[1:] - x0[:-1])**2)


def differential_signal_energy(cycle='70000', emitter=1, receiver=4, frequency=100):
    x0 = data(cycle='0', emitter=emitter, receiver=receiver, frequency=frequency)
    x = data(cycle=cycle, emitter=emitter, receiver=receiver, frequency=frequency)
    b = x0 / np.sqrt(np.sum(x0 * x0))
    d = x / np.sqrt(np.sum(x*x))
    return np.sum((b-d)**2)


def modified_mann_kendall(x, t):
    n = len(x)
    num = den = 0
    for i in range(n):
        for j in range(i+1, n):
            num += (t[j]-t[i])*(np.sign(x[j]-x[i]))
            den += t[j]-t[i]
            pass
        pass
    return abs(num / den)
