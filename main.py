import matplotlib.pyplot as plt
from shm_ugw_analysis.data_io.load_signals import (
    Signal,
    signal_collection,
    path_collection,
    frequency_collection,
    InvalidSignalError,
    allowed_cycles,
    allowed_signal_types,
    allowed_emitters,
    allowed_receivers,
    allowed_frequencies,
    all_paths
)
from shm_ugw_analysis.data_io.paths import PLOT_DIR

from shm_ugw_analysis.stats_processing.welch_psd_peaks import plot_signal_collection_psd_peaks, calculate_coherence

sb = Signal('0', 'received', 1, 4, 100)
s1 = Signal('1000', 'received', 1, 4, 100)
s2 = Signal('70000', 'received', 1, 4, 100)
# plot_psd_and_peaks(s, 4000)

sc = signal_collection(
    cycles=('30000',),
    signal_types=('excitation',),
    emitters=(1,),
    receivers=(4,),
    frequencies=(100, 120, 140, 160, 180,),
)

plot_signal_collection_psd_peaks(sc, bin_width=2000, file_label='changing_cycles')

coherence_b = calculate_coherence(sb, bin_width=2000)
coherence_s1 = calculate_coherence(s1, bin_width=2000)
coherence_s2 = calculate_coherence(s2, bin_width=2000)

fig, ax = plt.subplots(1, 1, figsize=(12, 10))
ax.plot(coherence_b[0], coherence_b[1], label='cycle 0')
ax.plot(coherence_s1[0], coherence_s1[1], label='cycle 1000')
ax.plot(coherence_s2[0], coherence_s2[1], label='cycle 70000')
ax.legend()
savepath = PLOT_DIR.joinpath('coherence.png')
plt.savefig(savepath, dpi=500)
