
from ..data_io.load_signals import load_data
import numpy as np 
from .basic_stats_method import basic_stats_summary



x = np.array([])
t = np.array([])
desc = np.array([])
folder = ""
filename = ""
full_path = ""

x, t, desc, folder, filename, full_path = load_data('1', 'excitation', 1, 4, 120)
print(basic_stats_summary(x))
print(full_path)
