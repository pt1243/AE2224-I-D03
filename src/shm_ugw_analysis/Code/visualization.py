import numpy as np
import matplotlib.pyplot as plt
from shm_ugw_analysis.data_io.signal import Signal


s = Signal('10000', 'received', 2, 5, 180)
x, t = s.x, s.t

start, end = np.searchsorted(t, -0.5e-5), np.searchsorted(t, 5e-5)
plt.plot(t[start:end], x[start:end])
#plt.plot(t, x)
plt.show()
