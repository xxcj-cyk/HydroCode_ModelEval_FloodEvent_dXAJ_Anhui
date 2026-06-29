import math

import numpy as np
from numba import jit

from hydromodels_pbm.utils.normalization import to_array_2d


# ---------------------------------------------------------------------------
# Production store
# ---------------------------------------------------------------------------


@jit(nopython=True)
def precip_store_in(s, precip_net, x1):
    n = x1 * (1.0 - (s / x1) ** 2) * np.tanh(precip_net / x1)
    d = 1.0 + (s / x1) * np.tanh(precip_net / x1)
    return n / d


@jit(nopython=True)
def evap_store_out(s, evap_net, x1):
    n = s * (2.0 - s / x1) * np.tanh(evap_net / x1)
    d = 1.0 + (1.0 - s / x1) * np.tanh(evap_net / x1)
    return n / d


@jit(nopython=True)
def percolation(store, x1):
    return store * (1.0 - (1.0 + (4.0 / 9.0 * store / x1) ** 4) ** -0.25)


@jit(nopython=True)
def production_step(p, e, x1, s):
    diff = p - e
    pn = diff if diff > 0.0 else 0.0
    en = -diff if diff < 0.0 else 0.0
    if s < 0.0:
        s = 0.0
    if s > x1:
        s = x1
    ps = precip_store_in(s, pn, x1)
    es = evap_store_out(s, en, x1)
    s = s - es + ps
    if s < 0.0:
        s = 0.0
    if s > x1:
        s = x1
    perc = percolation(s, x1)
    s = s - perc
    return perc + (pn - ps), es, s


# ---------------------------------------------------------------------------
# Unit hydrograph
# ---------------------------------------------------------------------------


@jit(nopython=True)
def s_curve_90(t, x4):
    if t <= 0:
        return 0.0
    if t < x4:
        return (t / x4) ** 2.5
    return 1.0


@jit(nopython=True)
def s_curve_10(t, x4):
    if t <= 0:
        return 0.0
    if t < x4:
        return 0.5 * (t / x4) ** 2.5
    if t < 2 * x4:
        return 1.0 - 0.5 * (2.0 - t / x4) ** 2.5
    return 1.0


@jit(nopython=True)
def unit_hydrograph_weights(x4, quickflow_90):
    if quickflow_90:
        n = int(math.ceil(x4))
    else:
        n = int(math.ceil(2.0 * x4))
    w = np.zeros(n)
    for t in range(1, n + 1):
        if quickflow_90:
            w[t - 1] = s_curve_90(t, x4) - s_curve_90(t - 1, x4)
        else:
            w[t - 1] = s_curve_10(t, x4) - s_curve_10(t - 1, x4)
    return w


@jit(nopython=True)
def convolve_truncated(x, w):
    n_time = x.shape[0]
    m = w.shape[0]
    out = np.zeros(n_time)
    for i in range(n_time):
        acc = 0.0
        for k in range(m):
            idx = i - k
            if idx >= 0:
                acc += x[idx] * w[k]
        out[i] = acc
    return out


# ---------------------------------------------------------------------------
# Routing store
# ---------------------------------------------------------------------------


@jit(nopython=True)
def route_store_step(q9, q1, x2, x3, x5, r):
    if r < 0.0:
        r = 0.0
    if r > x3:
        r = x3
    exchange = x2 * r / x3 - x2 * x5
    r = r + q9 + exchange
    if r < 0.0:
        r = 0.0
    qr = r * (1.0 - (1.0 + (r / x3) ** 4) ** -0.25)
    r = r - qr
    qd = q1 + exchange
    if qd < 0.0:
        qd = 0.0
    return qr + qd, r


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


@jit(nopython=True)
def simulate(p_and_e, x1, x2, x3, x4, x5):
    n_time, n_basin = p_and_e.shape[0], p_and_e.shape[1]
    prs = np.zeros((n_time, n_basin))
    ets = np.zeros((n_time, n_basin))
    s = 0.3 * x1.copy()
    for t in range(n_time):
        for j in range(n_basin):
            prs[t, j], ets[t, j], s[j] = production_step(
                p_and_e[t, j, 0], p_and_e[t, j, 1], x1[j], s[j]
            )

    q9 = np.zeros((n_time, n_basin))
    q1 = np.zeros((n_time, n_basin))
    for j in range(n_basin):
        w90 = unit_hydrograph_weights(x4[j], True)
        w10 = unit_hydrograph_weights(x4[j], False)
        q9[:, j] = convolve_truncated(prs[:, j], w90)
        q1[:, j] = convolve_truncated(prs[:, j], w10)

    qsim = np.zeros((n_time, n_basin))
    r = 0.5 * x3.copy()
    for t in range(n_time):
        for j in range(n_basin):
            qsim[t, j], r[j] = route_store_step(
                q9[t, j], q1[t, j], x2[j], x3[j], x5[j], r[j]
            )
    return qsim, ets


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------


def gr5j(p_and_e, parameters, warmup_length=365):
    params = to_array_2d(parameters)
    qsim, ets = simulate(
        p_and_e,
        params[:, 0],
        params[:, 1],
        params[:, 2],
        params[:, 3],
        params[:, 4],
    )
    w = max(0, int(warmup_length))
    return qsim[w:, :, np.newaxis], ets[w:, :]
