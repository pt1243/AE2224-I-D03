import numpy as np
import matplotlib.pyplot as plt
from shm_ugw_analysis.data_io.load_signals import Signal, all_paths, relevant_cycles

relevant_cycles = relevant_cycles[-7:]
average = np.zeros((6,7))

for p1, p2 in all_paths:
    results= []
    cycle_numbers = []
    for c in relevant_cycles:
        s = Signal(c, 'received', p1, p2, 180)
        x, t = s.x, s.t
        x=x-x.mean()


        start, end = np.searchsorted(t, 1e-5), np.searchsorted(t, 5e-5)
        cycle_numbers.append(int(c))
        results.append(np.sum(x[start:end]**2))
    
    average[p1-1] += np.array(results)
    #plt.subplot(9,2,counter)

    #plt.plot(cycle_numbers,results)
    #plt.title(f'{p1},{p2}')


for i in range(6):
    plt.subplot(6,1,i+1) 
    plt.plot([int(u) for u in relevant_cycles], average[i])
 
    


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