import math

import numpy as np
from numba import jit

from .XAJ import (
    calc_runoff_series,
    constrain_ki_kg,
    initial_storage_from_forcing,
    route_linear,
)
from hydromodels_pbm.utils.normalization import to_array_2d


# ---------------------------------------------------------------------------
# Unit hydrograph
# ---------------------------------------------------------------------------


@jit(nopython=True)
def gamma_uh_weights(a, theta, len_uh):
    aa = max(0.0, a) + 0.1
    th = max(0.0, theta) + 0.5
    w = np.empty(len_uh)
    denom = math.exp(math.lgamma(aa)) * (th**aa)
    acc = 0.0
    for i in range(len_uh):
        t = 0.5 + float(i)
        w[i] = (t ** (aa - 1.0)) * math.exp(-t / th) / denom
        acc += w[i]
    for i in range(len_uh):
        w[i] /= acc
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
# Routing
# ---------------------------------------------------------------------------


@jit(nopython=True)
def route_mz(runoff_im, rss, ris, rgs, a, theta, ci, cg, qi0, qg0, kernel_size):
    n_time, n_basin = ris.shape
    effective_len = kernel_size
    if effective_len > n_time:
        effective_len = n_time
    inp = runoff_im + rss
    qs_surface = np.zeros((n_time, n_basin))
    for j in range(n_basin):
        w = gamma_uh_weights(a[j], theta[j], effective_len)
        qs_surface[:, j] = convolve_truncated(inp[:, j], w)
    return route_linear(ris, rgs, qs_surface, ci, cg, qi0, qg0)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def simulate_xaj_mz(p_and_e, parameters, warmup_length=365, kernel_size=15):
    if p_and_e.shape[2] != 2:
        raise ValueError(
            'XAJ-mz expects drivers [time, basin, 2]: P, PET '
            f'(got feature dim {p_and_e.shape[2]})'
        )
    params = constrain_ki_kg(to_array_2d(parameters))
    k = params[:, 0]
    b = params[:, 1]
    im = params[:, 2]
    um = params[:, 3]
    lm = params[:, 4]
    dm = params[:, 5]
    c = params[:, 6]
    sm = params[:, 7]
    ex = params[:, 8]
    ki = params[:, 9]
    kg = params[:, 10]
    a = params[:, 11]
    theta = params[:, 12]
    ci = params[:, 13]
    cg = params[:, 14]

    wu, wl, wd, s = initial_storage_from_forcing(
        p_and_e, um, lm, dm, sm, warmup_length=warmup_length
    )
    n_basin = p_and_e.shape[1]
    fr = np.full(n_basin, 0.1)
    qi0 = np.full(n_basin, 0.1)
    qg0 = np.full(n_basin, 0.1)

    runoff_im, rss, ris, rgs, ets = calc_runoff_series(
        p_and_e, k, b, im, um, lm, dm, c, sm, ex, ki, kg, wu, wl, wd, s, fr
    )
    qs = route_mz(
        runoff_im, rss, ris, rgs, a, theta, ci, cg, qi0, qg0, kernel_size
    )
    return qs, ets


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------


def xaj_mz(p_and_e, parameters, warmup_length=365, kernel_size=15):
    qsim, ets = simulate_xaj_mz(
        p_and_e, parameters, warmup_length=warmup_length, kernel_size=kernel_size
    )
    w = max(0, int(warmup_length))
    return qsim[w:, :, np.newaxis], ets[w:, :]
