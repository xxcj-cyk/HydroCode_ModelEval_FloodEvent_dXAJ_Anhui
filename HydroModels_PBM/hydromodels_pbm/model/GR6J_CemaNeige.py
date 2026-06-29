import numpy as np
from numba import jit

from hydromodels_pbm.model.GR6J import simulate as gr6j_simulate
from hydromodels_pbm.utils.normalization import to_array_2d


# ---------------------------------------------------------------------------
# Solid precipitation
# ---------------------------------------------------------------------------


@jit(nopython=True)
def frac_solid_usace(temp):
    tmin = -1.0
    tmax = 3.0
    if temp <= tmin:
        return 1.0
    if temp >= tmax:
        return 0.0
    return 1.0 - (temp - tmin) / (tmax - tmin)


# ---------------------------------------------------------------------------
# CemaNeige
# ---------------------------------------------------------------------------


@jit(nopython=True)
def mean_annual_solid_precip(p, temp):
    n_time = p.shape[0]
    acc = 0.0
    for t in range(n_time):
        acc += frac_solid_usace(temp[t]) * p[t]
    if n_time == 0:
        return 0.0
    return acc / n_time * 365.25


@jit(nopython=True)
def cemaneige_step(p, temp, cn1, cn2, g, etg, gthreshold):
    frac = frac_solid_usace(temp)
    pliq = (1.0 - frac) * p
    g = g + frac * p
    etg = cn1 * etg + (1.0 - cn1) * temp
    if etg > 0.0:
        etg = 0.0
    if etg == 0.0 and temp > 0.0:
        pot_melt = cn2 * temp
        if pot_melt > g:
            pot_melt = g
    else:
        pot_melt = 0.0
    if g < gthreshold:
        gratio = g / gthreshold
    else:
        gratio = 1.0
    melt = (0.9 * gratio + 0.1) * pot_melt
    g = g - melt
    return pliq + melt, g, etg


@jit(nopython=True)
def cemaneige_series(p, temp, cn1, cn2, gthreshold):
    n_time, n_basin = p.shape[0], p.shape[1]
    pliq = np.zeros((n_time, n_basin))
    g = np.zeros(n_basin)
    etg = np.zeros(n_basin)
    for j in range(n_basin):
        for t in range(n_time):
            pliq[t, j], g[j], etg[j] = cemaneige_step(
                p[t, j], temp[t, j], cn1[j], cn2[j], g[j], etg[j], gthreshold[j]
            )
    return pliq


@jit(nopython=True)
def gthreshold_from_warmup(p, temp, warmup_length):
    n_time = p.shape[0]
    n_clim = int(warmup_length)
    if n_clim <= 0 or n_clim > n_time:
        n_clim = n_time
    n_basin = p.shape[1]
    out = np.zeros(n_basin)
    for j in range(n_basin):
        mean_solid = mean_annual_solid_precip(p[:n_clim, j], temp[:n_clim, j])
        out[j] = 0.9 * mean_solid
        if out[j] <= 0.0:
            out[j] = 1.0
    return out


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def gr6j_cemaneige(drivers, parameters, warmup_length=365):
    if drivers.shape[2] != 3:
        raise ValueError(
            'GR6J_CemaNeige expects drivers [time, basin, 3]: P, PET, T '
            f'(got feature dim {drivers.shape[2]})'
        )
    params = to_array_2d(parameters)
    p = drivers[:, :, 0]
    pet = drivers[:, :, 1]
    temp = drivers[:, :, 2]
    gthreshold = gthreshold_from_warmup(p, temp, warmup_length)
    pliq = cemaneige_series(p, temp, params[:, 6], params[:, 7], gthreshold)
    p_and_e = np.empty((drivers.shape[0], drivers.shape[1], 2), dtype=np.float64)
    p_and_e[:, :, 0] = pliq
    p_and_e[:, :, 1] = pet
    qsim, ets = gr6j_simulate(
        p_and_e,
        params[:, 0],
        params[:, 1],
        params[:, 2],
        params[:, 3],
        params[:, 4],
        params[:, 5],
    )
    w = max(0, int(warmup_length))
    return qsim[w:, :, np.newaxis], ets[w:, :]
