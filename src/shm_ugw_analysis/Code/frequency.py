import numpy as np
import matplotlib.pyplot as plt
from shm_ugw_analysis.data_io.signal import Signal


s = Signal('70000', 'received', 2, 5, 180)
x, t = s.x, s.t
start, end = np.searchsorted(t, -0.5e-5), np.searchsorted(t, 2.5e-5)
x = x[start:end]
x = x - np.mean(x)
x = x / np.std(x)

n, timestep = len(x), s.sample_interval
fourier = np.fft.fft(x) / n
frequencies = np.fft.fftfreq(n, d=timestep)

plt.plot(frequencies[:n//2], np.abs(fourier[:n//2]))
# plt.plot(np.abs(fourier[:n//2]))
plt.axvline(x=180e3, color='red', linestyle='--')
plt.axvline(x=280e3, color='blue', linestyle='--')
plt.xlim(1, 1e6)
plt.show()
