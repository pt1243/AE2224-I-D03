import numpy as np
from data_io.load import load_data


x, t, desc = load_data(
    cycle='0',
    signal_type='received',
    emitter=1,
    receiver=4,
    frequency=100
)
plt.plot(x,t)


plt.savefig('test.png')