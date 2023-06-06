def coherence(x_freq_baseline, x_freq_all_but_baseline):
    f, Cxy = coherence(x_freq_baseline, x_freq_all_but_baseline)
    return np.array([f, Cxy])
x_psd_welch, y_psd_welch = welch(s.x, fs, nperseg=nperseg)

psd_welch()

for i in allowed_cycles and not allowed_cycles = '0':
    coherence_array = coherence(y_psd_welch_magnitude_dB[0], y_psd_welch_magnitude_dB[i])

ax = plt.axes(projection='3d')
X = coherence_array[0]
Y = coherence_array[1]
Z: X, Y at different cycles 
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_title('surface')