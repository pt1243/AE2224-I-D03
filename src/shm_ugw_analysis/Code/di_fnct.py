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


def differential_curve_energy(cycle='70000', emitter=1, receiver=4, frequency=100):
    x0 = data(cycle='0', emitter=emitter, receiver=receiver, frequency=frequency)
    x = data(cycle=cycle, emitter=emitter, receiver=receiver, frequency=frequency)
    b = x0 - x
    return np.sum((b[1:] - b[:-1])**2) / np.sum((x0[1:] - x0[:-1])**2)
