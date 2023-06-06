# from __future__ import absolute_import

import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(script_dir))

import statsmodels.api as sm
from statsmodels.tsa.api import VAR
import pandas
from patsy import dmatrices
from ..data_io.load_signals import load_data
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.graphics.tsaplots as SMGT
import statsmodels.tsa.vector_ar.vecm as VECM

# Cointegration
# https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.coint.html#statsmodels.tsa.stattools.coint
# Note that method='aeg' representing the Augmented Engle-Granger Two-Step Method. Note that the y0 should be stationary time series in the shape of a 1D array - a baseline value that has limited variation. y1 may be an array containing cross-referenced values & yielding statistical indices on time-dependent cointegrated variations. 
# Johansen tests may be used to compare non-stationary time series - no stationary time series required as a baseline. statsmodels does not provide a Johansen test. 

def cointegrate_aeg(y0, y1):
    coint, pvalue, crit_value = sm.tsa.coint(y0, y1, trend='c', method='aeg', maxlag=None, autolag='aic', return_results=None)
    return [coint, pvalue, crit_value]

# Johansen
# Note that Vector Autoregressions (VAR) and Vector Error Correction Models (VECM) are used to simultaneously model and analyze several time series. Instead of comparing 2 time series, the vector format allows for higher order comparisons. 

# det_orderint = {-1, 0, 1} = {'no deterministic terms', 'constant term', 'linear trend'}
# det_orderint = -1
## There are no intercepts or trends in the cointegrated series and there are no deterministic trends in the levels of the data.
# det_orderint = 0
## There are intercepts in the cointegrated series and there are deterministic linear trends in the levels of the data. This is the default value.
# det_orderint = 1
## There are intercepts and linear trends in the cointegrated series and there are deterministic quadratic trends in the levels of the data.

def cointegrate_johansen(endog, det_order, k_ar_diff):
    out = VECM.coint_johansen(endog, det_order, k_ar_diff)
    return out

# Output of Johansen integration is extensive. See:
# https://www.statsmodels.org/dev/generated/statsmodels.tsa.vector_ar.vecm.JohansenTestResult.html#statsmodels.tsa.vector_ar.vecm.JohansenTestResult

# Skewness
# y: array
def skewness(y, axis):
    out = sm.stats.stattools.robust_skewness(y, axis)
    return out

# Kurtosis
# y: array
def kurtosis(y, axis):
    out = sm.stats.stattools.robust_kurtosis(y, axis)
    return out

# Autocorrelation Function - Time Series - FFT = True
def autocorrelation_fft(x):
    out = sm.tsa.stattools.acf(x, adjusted=False, nlags=None, qstat=False, fft=True, alpha=None, bartlett_confint=True, missing='none')
    return out 

def autocorrelation_fft_plot(x):
    out = SMGT.plot_acf(x, adjusted=False, nlags=None, qstat=False, fft=True, alpha=None, bartlett_confint=True, missing='none')
    return out

### NOTES ON METHODOLOGY ###
# We must find a way to process signal data to 'smoothen' it before applying certain statistical markers. For example, applying skewness or kurtosis on erratic signal data will not reveal much about the behaviour about the data even though a value will be provided by the function.
# Identifying change for each signal path between healthy and damaged state will yield change in the time-domain. Cointegration may be used as a basis to identifying these changes. 
# We may want to represent global variation in health state - we would need to condense all signal paths into one net marker that may be projected through each cycle. Each signal may be scaled/realigned to amplify features before processing them over the cycle axis. 
# Note that we also have 5 different frequency bands (5, 6, 7, 8, 9)x20kHz that may (should) reveal new features. 

### PLOTTING EXPERIMENTS ###

# Extracting signals 

cycles = ['AI', '0', '1', '1000', '10000', '20000', '30000', '40000', '50000', '60000', '70000', 'Healthy']
signal_types = ['excitation', 'received']
signal_type_subscript = ['e', 'r']
#emitters = [i for i in range(1, 7)]
#receivers = emitters
#erpath = [[1, 4], [1, 5], [1, 6]]
frequencies = [100, 120, 140, 160, 180]

data = {}
'''
for i in range(1, len(cycles)-1):
    for j in range(len(signal_types)):
        for k in range(len(frequencies)):
            name = f'_{cycles[i]}_{signal_type_subscript[j]}_{1}_{4}_{frequencies[k]}'
            data[str(x)+name, str(t)+name, str(desc)+name] = load_data(
            cycle=cycles[i],
            signal_type=signal_types[j],
            emitter=1,
            receiver=4,
            frequency=frequencies[k]
    )
'''
'''
t_0_e_1_4_100, x_0_e_1_4_100, desc_0_e_1_4_100 = load_data(
    cycle='0',
    signal_type='excitation',
    emitter=1,
    receiver=4,
    frequency=100
)

t_0_r_1_4_100, x_0_r_1_4_100, desc_0_r_1_4_100 = load_data(
    cycle='0',
    signal_type='received',
    emitter=1,
    receiver=4,
    frequency=100
)

t_0_e_1_5_100, x_0_e_1_5_100, desc_0_e_1_5_100 = load_data(
    cycle='0',
    signal_type='excitation',
    emitter=1,
    receiver=5,
    frequency=100
)

t_0_r_1_5_100, x_0_r_1_5_100, desc_0_r_1_5_100 = load_data(
    cycle='0',
    signal_type='received',
    emitter=1,
    receiver=5,
    frequency=100
)

t_0_e_1_6_100, x_0_e_1_6_100, desc_0_e_1_6_100 = load_data(
    cycle='0',
    signal_type='excitation',
    emitter=1,
    receiver=6,
    frequency=100
)

t_0_r_1_6_100, x_0_r_1_6_100, desc_0_r_1_6_100 = load_data(
    cycle='0',
    signal_type='received',
    emitter=1,
    receiver=6,
    frequency=100
)

t_1, x_1, _ = load_data(
    cycle='0',
    signal_type='received',
    emitter=1,
    receiver=6,
    frequency=140
)

t_2, x_2, _ = load_data(
    cycle='70000',
    signal_type='received',
    emitter=1,
    receiver=6,
    frequency=140
)

matrix = np.empty
index = np.empty
for i in range(1, len(cycles)-1):
    for j in range(len(signal_types)):
        for k in range(len(frequencies)):
            name = f'_{cycles[i]}_{signal_type_subscript[j]}_{1}_{4}_{frequencies[k]}'
            data[str(x)+name, str(t)+name, str(desc)+name] = load_data(
                cycle=cycles[i],
                signal_type=signal_types[j],
                emitter=1,
                receiver=4,
                frequency=frequencies[k]
                )

            out = cointegrate_aeg(x_{cycles[0]}_{signal_type_subscript[j]}_1_4_{frequencies[k]}, x_{cycles[i]}_{signal_type_subscript[j]}_1_4_{frequencies[k]})
            index = [f'{cycles[i]}_{signal_type_subscript[j]}_1_4_{frequencies[k]}']
            np.append(matrix, out)
            print(arr)
            print(matrix)

# Plotting excited-received signals on top of one another to understand interaction

a_0_e_1_4_100 = np.where(t_0_e_1_4_100 > 0.0)[0][0]
b_0_e_1_4_100 = np.where(t_0_e_1_4_100 > 0.00002)[0][0]
a_0_r_1_4_100 = np.where(t_0_r_1_4_100 > 0.0)[0][0]
b_0_r_1_4_100 = np.where(t_0_r_1_4_100 > 0.00002)[0][0]

a_0_e_1_5_100 = np.where(t_0_e_1_5_100 > 0.0)[0][0]
b_0_e_1_5_100 = np.where(t_0_e_1_5_100 > 0.00002)[0][0]
a_0_r_1_5_100 = np.where(t_0_r_1_5_100 > 0.0)[0][0]
b_0_r_1_5_100 = np.where(t_0_r_1_5_100 > 0.00002)[0][0]

a_0_e_1_6_100 = np.where(t_0_e_1_6_100 > 0.0)[0][0]
b_0_e_1_6_100 = np.where(t_0_e_1_6_100 > 0.00002)[0][0]
a_0_r_1_6_100 = np.where(t_0_r_1_6_100 > 0.0)[0][0]
b_0_r_1_6_100 = np.where(t_0_r_1_6_100 > 0.00002)[0][0]

#cointegrate_johansen(x_0_e_1_4_100[a_0_e_1_4_100:b_0_e_1_4_100], det_order=0, k_ar_diff=5)

fig, axs = plt.subplots(3, 1, figsize=(8, 10))

axs[0].plot(t_0_e_1_4_100[a_0_e_1_4_100:b_0_e_1_4_100], x_0_e_1_4_100[a_0_e_1_4_100:b_0_e_1_4_100], label='EXCITED 1-4 100kHz')
axs[0].plot(t_0_r_1_4_100[a_0_r_1_4_100:b_0_r_1_4_100], x_0_r_1_4_100[a_0_r_1_4_100:b_0_r_1_4_100], label='RECEIVED 1-4 100kHz')
axs[0].legend()
axs[1].plot(t_0_e_1_5_100[a_0_e_1_5_100:b_0_e_1_5_100], x_0_e_1_5_100[a_0_e_1_5_100:b_0_e_1_5_100], label='EXCITED 1-5 100kHz')
axs[1].plot(t_0_r_1_5_100[a_0_r_1_5_100:b_0_r_1_5_100], x_0_r_1_5_100[a_0_r_1_5_100:b_0_r_1_5_100], label='RECEIVED 1-5 100kHz')
axs[1].legend()
axs[2].plot(t_0_e_1_6_100[a_0_e_1_6_100:b_0_e_1_6_100], x_0_e_1_6_100[a_0_e_1_6_100:b_0_e_1_6_100], label='EXCITED 1-6 100kHz')
axs[2].plot(t_0_r_1_6_100[a_0_r_1_6_100:b_0_r_1_6_100], x_0_r_1_6_100[a_0_r_1_6_100:b_0_r_1_6_100], label='RECEIVED 1-6 100kHz')
axs[2].legend()

# Set titles for the subplots
#axs[0].set_title('Subplot 1')
#axs[1].set_title('Subplot 2')
#axs[2].set_title('Subplot 3')

# Set overall title for the figure
fig.suptitle('1-4, 1-5, 1-6 E&R Paths at 100kHz')

# Show the figure
#plt.show()

# Autocorrelation Function Plot
fig, axs = plt.subplots(3, 2, figsize=(8, 12))
SMGT.plot_acf(x_0_e_1_4_100[a_0_e_1_4_100:b_0_e_1_4_100], ax=axs[0,0])
axs[0,0].set_title('EXCITED 1-4 100kHz')
SMGT.plot_acf(x_0_r_1_4_100[a_0_r_1_4_100:b_0_r_1_4_100], ax=axs[0,1])
axs[0,1].set_title('RECEIVED 1-4 100kHz')
SMGT.plot_acf(x_0_e_1_5_100[a_0_e_1_5_100:b_0_e_1_5_100], ax=axs[1,0])
axs[1,0].set_title('EXCITED 1-5 100kHz')
SMGT.plot_acf(x_0_r_1_5_100[a_0_r_1_5_100:b_0_r_1_5_100], ax=axs[1,1])
axs[1,1].set_title('RECEIVED 1-5 100kHz')
SMGT.plot_acf(x_0_e_1_6_100[a_0_e_1_6_100:b_0_e_1_6_100], ax=axs[2,0])
axs[2,0].set_title('EXCITED 1-6 100kHz')
SMGT.plot_acf(x_0_r_1_6_100[a_0_r_1_6_100:b_0_r_1_6_100], ax=axs[2,1])
axs[2,1].set_title('RECEIVED 1-6 100kHz')

#plt.show()
'''
