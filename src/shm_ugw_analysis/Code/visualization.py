import numpy as np
import matplotlib.pyplot as plt
from shm_ugw_analysis.data_io.load_signals import Signal


s = Signal('70000', 'received', 2, 5, 180)
x, t = s.x, s.t
x = x - x.mean()

s0 = Signal('0', 'received', 2, 5, 180)
x0, t0 = s0.x, s0.t
x0 = x0 - x0.mean()

xr = x - x0

start, end = np.searchsorted(t, -0.5e-5), np.searchsorted(t, 20e-5)
plt.plot(t[start:end], x[start:end])
plt.axvline(x=t[start], color='blue', linestyle='--')
plt.axvline(x=t[start+500], color='blue', linestyle='--')
#plt.plot(t, x)
#plt.plot(t, xr)
#plt.plot(t, x0)
plt.show()
