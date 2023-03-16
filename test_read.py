#import npz files stored in the same directory
import numpy as np

# Load npz file
data = np.load('L2_S2_cycle_0\C1_mono_1_saine_emetteur_1_recepteur_2_excitation_5_00000.npz')
print(data)
# Access the data in the npz file
array1 = data['arr_0']
array2 = data['arr_1']