from shm_ugw_analysis.data_io.load_signals import load_data
import matplotlib.pyplot as plt

xtab = []
ttab = []

x, t, desc = load_data(
    cycle='0',
    signal_type='received',
    emitter=1,
    receiver=4,
    frequency=100)

xtab.append(load_data[1])
ttab.append(load_data[2])

plt.plot(xtab,ttab)
plt.show()