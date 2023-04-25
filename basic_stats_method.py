import numpy as np

import pandas


def basic_stats_summary(y):
    mean = np.mean(y)
    mean_square = np.mean(y**2)
    rms = np.sqrt(mean_square)
    std = np.std(y)
    var = std ** 2
    peak_amp = max(y)
    crest_factor = (peak_amp) / (rms)
    K_factor = peak_amp * rms
    clearance_factor = (peak_amp) / (mean_square)
    impulse_factor = (peak_amp) / (mean_square)
    shape_factor = (rms)/(np.mean(abs(y)),axis)
    return mean, mean_square, rms, std, var, peak_amp, crest_factor, K_factor, clearance_factor, impulse_factor, shape_factor


emitter = np.array([1,2,3])
receiver = np.array([4,5,6])
frequency = np.array([100, 120, 140, 160, 180])
cycles = np.array(['0', '1', '1000', '10000', '20000', '30000', '40000', '50000', '60000', '70000', 'Healthy', 'AI'])
