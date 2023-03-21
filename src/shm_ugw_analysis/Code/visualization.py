import numpy as np
import matplotlib.pyplot as plt
from shm_ugw_analysis.data_io.load import load_data

x, t, desc = load_data(cycle='1', signal_type='received', emitter=1, receiver=6, frequency=100)

start, end = np.searchsorted(t, -0.5e-5), np.searchsorted(t, 2.5e-5)
plt.plot(t[start:end], x[start:end])
plt.show()
