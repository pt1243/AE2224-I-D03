import statsmodels.api as sm
#from statsmodels.tsa.api import VAR
import pandas
from patsy import dmatrices

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
    out = sm.tsa.VECM.coint_johansen(endog, det_order, k_ar_diff)
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

### NOTES ON METHODOLOGY ###
# We must find a way to process signal data to 'smoothen' it before applying certain statistical markers. For example, applying skewness or kurtosis on erratic signal data will not reveal much about the behaviour about the data even though a value will be provided by the function.
# Identifying change for each signal path between healthy and damaged state will yield change in the time-domain. Cointegration may be used as a basis to identifying these changes. 
# We may want to represent global variation in health state - we would need to condense all signal paths into one net marker that may be projected through each cycle. Each signal may be scaled/realigned to amplify features before processing them over the cycle axis. 
# Note that we also have 5 different frequency bands (5, 6, 7, 8, 9)x20kHz that may (should) reveal new features. 