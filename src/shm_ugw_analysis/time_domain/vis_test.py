import numpy as np
import matplotlib.pyplot as plt
from shm_ugw_analysis.data_io.load_signals import Signal


exc_freq = 100
s = Signal('70000', 'received', 2, 5, exc_freq)
x, t = s.x, s.t
x = x - x.mean()

start, end = np.searchsorted(t, 0e-5), np.searchsorted(t, 10e-5)
plt.axvline(x=2e-3/exc_freq, color='blue', linestyle='--')
plt.axvline(x=2e-3/exc_freq + 4e-5, color='blue', linestyle='--')

plt.plot(t[start:end], x[start:end])
plt.show()
print(2e-3/exc_freq)
