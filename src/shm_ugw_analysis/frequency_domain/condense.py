from ..data_io.load_signals import load_data
import numpy as np
import matplotlib.pyplot as plt

#Rendered using Chat-GPT 

cycle = '0'
signal_type = ['excitation', 'received']
emitter = 1
receiver = [4, 5, 6]
frequency = 100

data = {}

for s in signal_type:
    for r in receiver:
        name = f't_{cycle}_{s[0]}_{emitter}_{r}_{frequency}'
        data[name] = load_data(cycle=cycle, signal_type=s, emitter=emitter, receiver=r, frequency=frequency)

fig, axs = plt.subplots(len(receiver), len(signal_type), figsize=(10, 8))

for i, r in enumerate(receiver):
    for j, s in enumerate(signal_type):
        t_name = f't_{cycle}_{s[0]}_{emitter}_{r}_{frequency}'
        x_name = f'x_{cycle}_{s[0]}_{emitter}_{r}_{frequency}'
        a_name = f'a_{cycle}_{s[0]}_{emitter}_{r}_{frequency}'
        b_name = f'b_{cycle}_{s[0]}_{emitter}_{r}_{frequency}'

        a = np.where(data[t_name] > 0.0)[0][0]
        b = np.where(data[t_name] > 0.00002)[0][0]

        axs[i,j].plot(data[t_name][a:b], data[x_name][a:b], label=f'{s[0].upper()}-{emitter}-{r}-{frequency}kHz')
        axs[i,j].legend()

plt.show()