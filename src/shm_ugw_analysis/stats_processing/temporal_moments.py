# from functools import lru_cache, wraps

import numpy as np

from shm_ugw_analysis.data_io.load_signals import Signal


def _k_order_temporal_moment(s: Signal, k: int, t_s: float):
    """kth order temporal moment, M_k(t_s)"""
    n = s.x.shape[0]
    x_i_1 = s.x[1:]
    x_i = s.x[:-1]
    x_arr = x_i ** 2 + x_i_1 ** 2
    delta_t = s.sample_interval
    if k == 0:
        delta_t_arr = np.ones(n - 1)
    else:
        delta_t_arr = ((np.arange(n - 1) + 1.5) * delta_t - t_s) ** k
    return 0.5 * np.sum(delta_t_arr * delta_t * x_arr)


def E(s: Signal):
    """Energy, defined as M_0(0)"""
    return _k_order_temporal_moment(s, 0, 0)


def T(s: Signal):
    """Central time T, defined as M_1(0) / E"""
    return _k_order_temporal_moment(s, 1, 0) / E(s)


def D(s: Signal):
    """Root mean square duration D, defined as sqrt(M_2(T) / E)"""
    return np.sqrt(_k_order_temporal_moment(s, 2, T(s)) / E(s))


def A_e(s: Signal):
    """Root energy amplitude A_e, defined as sqrt(E/D)"""
    return np.sqrt(E(s) / D(s))


def S_t_3(s: Signal):
    """Central skewness S_t^3, defined as M_3(T)/E"""
    return _k_order_temporal_moment(s, 3, T(s)) / E(s)


def S(s: Signal):
    """Normalized skewness S, defined as S_t/D"""
    return np.cbrt(S_t_3(s)) / D(s)


def K_t_4(s: Signal):
    """Central kurtosis K_t^4, defined as M_4(T)/E"""
    return _k_order_temporal_moment(s, 4, T(s)) / E(s)


def K(s: Signal):
    """Normalized kurtosis K, defined as K_t/D"""
    return K_t_4(s) ** (1/4) / D(s)
