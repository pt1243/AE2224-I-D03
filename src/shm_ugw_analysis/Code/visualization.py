import numpy as np
import matplotlib.pyplot as plt
from shm_ugw_analysis.data_io.load import load_data

x, t, desc = load_data(cycle='0', signal_type='received', emitter=2, receiver=5, frequency=100)

print(np.shape(x), np.shape(t), desc)
plt.plot(x[2475:2625], t[2475:2625])
plt.show()
