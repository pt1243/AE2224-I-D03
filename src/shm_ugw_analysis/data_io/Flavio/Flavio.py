import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import os
from shm_ugw_analysis.data_io.load import load_data

#healthy_path = "C:\\Users\\Gebruiker\\Desktop\\TU DELFT\\Year 2\\Project 2nd semester\\npz_files\\npz_files\\L2_S2_cycle_60000"
#healthy_files = [f for f in os.listdir(healthy_path) if f.endswith('.npz')]
#healthy_lists = [[] for _ in range(len(healthy_files))]

#for i, healthy_file in enumerate(healthy_files):
    #healthy_data = np.load(os.path.join(healthy_path, healthy_file))
    #for array_name in healthy_data.files:
        #healthy_lists[i].append(healthy_data[array_name])

#x = healthy_lists[50][0]
#y = healthy_lists[50][1]

t, y, desc = load_data('0', 'received', 1, 4, 100)

#plt.xlim(0,0.00005)
#plt.tlim(-10,10)


plt.plot(t, y)
plt.xlim(0,0.00005)
plt.ylim(-0.1,0.1)
plt.xlabel('t')
plt.ylabel('y')
plt.title('godo')
plt.savefig('test1.png')