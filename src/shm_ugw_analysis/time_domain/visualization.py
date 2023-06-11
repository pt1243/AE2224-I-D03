import numpy as np
import matplotlib.pyplot as plt
from shm_ugw_analysis.data_io.load_signals import Signal


s = Signal('70000', 'received', 2, 5, 180)
x, t = s.x, s.t
x = x - x.mean()

s1 = Signal('10000', 'received', 2, 5, 180)
x1, t1 = s1.x, s1.t
x1 = x1 - x1.mean()

s3 = Signal('40000', 'received', 2, 5, 180)
x3, t3 = s3.x, s3.t
x3 = x3 - x3.mean()

#xr = x - x0

start, end = np.searchsorted(t, 2e-3/180), np.searchsorted(t, 2e-3/180 + 4e-5)
plt.figure(figsize=[1.5*6.4, 4.8*1.5])
plt.xlabel("Time [s]")
plt.ylabel("Amplitude [V]")
#plt.axvline(x=t[start], color='blue', linestyle='--')
#plt.axvline(x=t[start+500], color='blue', linestyle='--')
plt.plot(t[start:end], x1[start:end])
plt.plot(t[start:end], x3[start:end])
plt.plot(t[start:end], x[start:end])
plt.legend(["10,000 cycles", "40,000 cycles", "70,000 cycles"])
#plt.plot(t, x0)
plt.savefig("plots/cycles", dpi=500)
plt.show()
