
from src.shm_ugw_analysis.data_io.load import load_data
import numpy as np 
import basic_stats_method as bm



x = np.array([])
t = np.array([])

x, t = load_data('1', 'excitation', 1, 4, 120)
bm.basic_stats_summary(x)
