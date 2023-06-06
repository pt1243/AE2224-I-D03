import src.shm_ugw_analysis.data_io.load_signals as ssumdl
import numpy as np 
import basic_stats_method as bm


x = np.array([])
t = np.array([])
desc = np.array([])
folder = ""
filename = ""
full_path = ""

cycle = input("Enter desired cycle:")
signal_type = input("Enter desired signal type: ")
emitter = int(input("Enter desired emitter: "))
receiver = int(input("Enter desired receiver: "))
frequency = int(input("Enter desired frequency: "))

x, t, desc, folder, filename, full_path = ssumdl.load_data(cycle, signal_type, emitter, receiver, frequency)
mean, mean_square, rms, std, var, peak_amp, crest_factor, K_factor, clearance_factor, impulse_factor, shape_factor = bm.basic_stats_summary(x)
print("mean: " + str(mean) + "\nmean_square: " + str(mean_square) +  "\nrms: " + str(rms) + "\nstd: " + str(std) + "\nvar: " + str(var) + "\npeak_amp" + str(peak_amp) + "\ncrest_factor" + str(crest_factor) + "\nK_factor" + str(K_factor) + "\nclearance_factor: " + str(clearance_factor) + "\nimpulse_factor: " + str(impulse_factor) + "\nshape_factor: " + str(shape_factor))
