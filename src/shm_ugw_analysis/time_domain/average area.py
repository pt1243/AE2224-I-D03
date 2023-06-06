import numpy as np
import matplotlib.pyplot as plt
from ..data_io.load_signals import Signal, all_paths, relevant_cycles, allowed_frequencies

relevant_cycles = relevant_cycles[-7:]
average = np.zeros((5,7))

for i in allowed_frequencies:
    for p1, p2 in all_paths:
        results= []
        for c in relevant_cycles:
            s = Signal(c, 'received', p1, p2, i)
            x, t = s.x, s.t
            x=x-x.mean()


            start, end = np.searchsorted(t, 1e-5), np.searchsorted(t, 5e-5)
            results.append(np.sum(x[start:end]**2))
        
        average[int((i-100)/20)] += np.array(results)
    


for i in range(5):
  # plt.subplot(6,1,i+1) 
  plt.plot([int(u) for u in relevant_cycles], average[i]/6)
plt.legend(['Frequency 100', 'Frequency 120', 'Frequency 140', 'Frequency 160', 'Frequency 180'])
plt.xlabel("Cycle number")
plt.ylabel("Magnitude")



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