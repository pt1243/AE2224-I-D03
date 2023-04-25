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
    relevant_cycles,
    all_paths,
)
from shm_ugw_analysis.data_io.paths import PLOT_DIR

from shm_ugw_analysis.stats_processing.welch_psd_peaks import plot_signal_collection_psd_peaks, calculate_coherence, plot_coherence

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

sc_coherence = signal_collection(
    cycles=relevant_cycles,
    signal_types=('excitation',),
    emitters=(1,),
    receivers=(4,),
    frequencies=(160,)
)

plot_coherence(sc_coherence, bin_width=2000, sigma=1.5)
