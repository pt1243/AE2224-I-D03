import matplotlib.pyplot as plt
import numpy as np
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

from shm_ugw_analysis.frequency_domain.welch_psd_peaks import plot_signal_collection_psd_peaks, calculate_coherence, plot_coherence, plot_coherence_3d, butter_lowpass, our_fft, plot_signal_collection_fft, real_find_peaks

from shm_ugw_analysis.frequency_domain.peaks_damage_indices import search_peaks_arrays, generate_magnitude_array, plot_DI, plot_all_DIs

plot_DI('maximum', 1)
plot_DI('maximum', 2)
plot_DI('maximum', 3)
plot_DI('minimum', 1)
plot_DI('minimum', 2)
plot_DI('minimum', 3)
plot_DI('minimum', 4)
