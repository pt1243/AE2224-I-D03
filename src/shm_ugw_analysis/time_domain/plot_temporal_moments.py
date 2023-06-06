import matplotlib.pyplot as plt

from .temporal_moments import E, T, D, A_e, S_t_3, S, K_t_4, K
from ..data_io.load_signals import signal_collection
from ..data_io.paths import ROOT_DIR

sc = signal_collection(
    cycles=['0', '1', '1000', '10000', '20000', '30000', '40000', '50000', '60000', '70000'],
    signal_types=['received'],
    emitters=[1],
    receivers=[6],
    frequencies=[100]
)

fig, axs = plt.subplots(4, 2, figsize=(20, 20))
E_list, T_list, D_list, A_e_list, S_t_3_list, S_list, K_t_4_list, K_list = [], [], [], [], [], [], [], []
cycle_list = []

for s in sc:
    cycle_list.append(int(s.cycle))
    E_list.append(E(s))
    T_list.append(T(s))
    D_list.append(D(s))
    A_e_list.append(A_e(s))
    S_t_3_list.append(S_t_3(s))
    S_list.append(S(s))
    K_t_4_list.append(K_t_4(s))
    K_list.append(K(s))

axs[0, 0].plot(cycle_list, E_list)
axs[0, 1].plot(cycle_list, T_list)
axs[1, 0].plot(cycle_list, D_list)
axs[1, 1].plot(cycle_list, A_e_list)
axs[2, 0].plot(cycle_list, S_t_3_list)
axs[2, 1].plot(cycle_list, S_list)
axs[3, 0].plot(cycle_list, K_t_4_list)
axs[3, 1].plot(cycle_list, K_list)


axs[0, 0].set_ylabel('Energy')
axs[0, 1].set_ylabel('Central time')
axs[1, 0].set_ylabel('RMS duration')
axs[1, 1].set_ylabel('Root energy amplitude')
axs[2, 0].set_ylabel('Central skewness')
axs[2, 1].set_ylabel('Normalized skewness')
axs[3, 0].set_ylabel('Central kurtosis')
axs[3, 1].set_ylabel('Normalized kurtosis')
for i in range(4):
    for j in range(2):
        axs[i, j].set_xlabel('Cycle number')

filename = ROOT_DIR.joinpath('src', 'shm_ugw_analysis', 'statistical_processing', 'temporal_moments_plot.png')

plt.savefig(filename, dpi=500, bbox_inches='tight')
