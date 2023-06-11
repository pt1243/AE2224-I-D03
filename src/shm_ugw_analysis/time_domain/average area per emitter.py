import numpy as np
import matplotlib.pyplot as plt
from shm_ugw_analysis.data_io.load_signals import Signal, all_paths, relevant_cycles, allowed_frequencies

relevant_cycles = relevant_cycles[-7:]
average = np.zeros((5,7))
print(all_paths)
for i in allowed_frequencies:
    for p2 in range(4,7):
        results= []
        for c in relevant_cycles:
            s = Signal(c, 'received', 1, p2, i)
            x, t = s.x, s.t
            x=x-x.mean()

            start, end = np.searchsorted(t, 2e-3/i), np.searchsorted(t, 2e-3/i + 4e-5)
            results.append(np.sum(x[start:end]**2))
        
        average[int((i-100)/20)] += np.array(results)
    

markers = ['+', 'o', (5,2), '>', (4,0), (5,1)]
for i in range(5):
  # plt.subplot(6,1,i+1) 
  plt.plot([int(u) for u in relevant_cycles], average[i], marker=markers[i])
plt.legend(['100 kHz', '120 kHz', '140 kHz', '160 kHz', '180 kHz'],loc="upper left")
plt.xlabel("Cycle number")
plt.ylabel("Energy [$V^2$]")
plt.savefig("plots/Averaged Paths per Frequency", dpi=500)


#for c in range(10000,70001,10000):
    #s = Signal(str(c), 'received', 2, 6, 180)
    #x, t = s.x, s.t
    #x=x-x.mean()


    #start, end = np.searchsorted(t, 0e-5), np.searchsorted(t, 5e-5)

    #print(np.sum(x[start:end]**2))

#plt.plot(c,)
#plt.plot(t[start:end], x[start:end])
#plt.plot(t, x)
#plt.plot(t[start:], x[start:])
plt.show()