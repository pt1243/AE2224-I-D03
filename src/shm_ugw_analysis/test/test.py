import numpy as np
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt 
from data_io.load import load_data

# Load npz file
data = np.load('L2_S2_cycle_0\C1_mono_1_saine_emetteur_1_recepteur_2_excitation_5_00000.npz')
print(data)
# Access the data in the npz file
array1 = data['arr_0']
array2 = data['arr_1']

x, t, desc = load_data(
    cycle='0',
    signal_type='received',
    emitter=1,
    receiver=4,
    frequency=100
)

print(x)

