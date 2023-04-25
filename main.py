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

from shm_ugw_analysis.stats_processing.welch_psd_peaks import plot_signal_collection_psd_peaks

# s = Signal('20000', 'received', 1, 4, 100)
# plot_psd_and_peaks(s, 4000)

sc = signal_collection(
    cycles=allowed_cycles,
    signal_types=('received',),
    emitters=(1,),
    receivers=(4,),
    frequencies=(100,),
)

plot_signal_collection_psd_peaks(sc, bin_width=2000, file_label='changing_cycles')
