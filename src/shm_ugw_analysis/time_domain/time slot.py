import numpy as np
import matplotlib.pyplot as plt
from ..data_io.load_signals import Signal




for c in range(10000,70001,10000):
    s = Signal(str(c), 'received', 2, 5, 180)
    x, t = s.x, s.t
    x=x-x.mean()


    start, end = np.searchsorted(t, 0e-5), np.searchsorted(t, 5e-5)
    #plt.plot(t[start:end], x[start:end])
    #plt.plot(t, x)
    #plt.plot(t[start:], x[start:])

    #max_values=[]
    i_values=0

    for i in range(start,len(x)) :
        max_value = np.max(np.abs(x[i:i+500]))
        if max_value<0.01:
            i_values=i 
            break

        #max_values.append(max_value)
    #plt.plot(t[start:], max_values)
    print(t[i_values])
    #plt.axvline(x=t[i_values], color='red', linestyle='--')
    #plt.show()
