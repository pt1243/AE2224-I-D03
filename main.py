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

from shm_ugw_analysis.stats_processing.welch_psd_peaks import find_psd_peaks, psd_welch, create_and_save_fig, plot_psd_and_peaks

s = Signal('20000', 'received', 1, 4, 100)
plot_psd_and_peaks(s, 5000)
