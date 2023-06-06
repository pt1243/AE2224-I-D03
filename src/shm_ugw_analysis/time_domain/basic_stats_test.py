
import src.shm_ugw_analysis.data_io.load_signals as ssumdl
import numpy as np 
import basic_stats_method as bm



x = np.array([])
t = np.array([])
desc = np.array([])
folder = ""
filename = ""
full_path = ""

x, t, desc, folder, filename, full_path = ssumdl.load_data('1', 'excitation', 1, 4, 120)
print(bm.basic_stats_summary(x))
print(full_path)
