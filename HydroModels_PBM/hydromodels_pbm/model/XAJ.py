import numpy as np
from numba import jit

from hydromodels_pbm.utils.normalization import to_array_2d


# ---------------------------------------------------------------------------
# Initial storage
# ---------------------------------------------------------------------------


def initial_storage_from_forcing(p_and_e, um, lm, dm, sm, warmup_length=365):
    prcp = np.maximum(p_and_e[:, :, 0], 0.0)
    pet = np.maximum(p_and_e[:, :, 1], 0.0)
    n_time = prcp.shape[0]
    n = max(1, min(int(warmup_length), n_time))
    p_bar = np.mean(prcp[:n], axis=0)
    e_bar = np.mean(pet[:n], axis=0)
    moisture = np.clip(p_bar / np.maximum(e_bar, 1e-6), 0.5, 1.5)
    wu = np.clip(0.5 * um * moisture, 0.15 * um, 0.85 * um)
    wl = np.clip(0.5 * lm * moisture, 0.15 * lm, 0.85 * lm)
    wd = 0.5 * dm
    s = np.clip(0.5 * sm * moisture, 0.15 * sm, 0.85 * sm)
    return wu, wl, wd, s


def constrain_ki_kg(params, ki_idx=9, kg_idx=10, max_sum=0.99):
    arr = np.asarray(params, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    s = arr[:, ki_idx] + arr[:, kg_idx]
    over = s > max_sum
    if np.any(over):
        scale = np.where(over, max_sum / s, 1.0)
        arr = arr.copy()
        arr[:, ki_idx] = arr[:, ki_idx] * scale
        arr[:, kg_idx] = arr[:, kg_idx] * scale
    return arr


# ---------------------------------------------------------------------------
# Runoff
# ---------------------------------------------------------------------------


@jit(nopython=True)
def calc_runoff_series(
    p_and_e,
    k,
    b,
    im,
    um,
    lm,
    dm,
    c,
    sm,
    ex,
    ki,
    kg,
    wu0,
    wl0,
    wd0,
    s0,
    fr0,
):
    n_time, n_basin, _ = p_and_e.shape
    pe_eps = 1e-6
    fr_max = 1.0
    runoff_im = np.zeros((n_time, n_basin))
    rss = np.zeros((n_time, n_basin))
    ris = np.zeros((n_time, n_basin))
    rgs = np.zeros((n_time, n_basin))
    ets = np.zeros((n_time, n_basin))
    wu = wu0.copy()
    wl = wl0.copy()
    wd = wd0.copy()
    s = s0.copy()
    fr = fr0.copy()

    wm = np.empty(n_basin)
    w_cap = np.empty(n_basin)
    for j in range(n_basin):
        wm[j] = um[j] + lm[j] + dm[j]
        if wm[j] < 1e-5:
            wm[j] = 1e-5
        w_cap[j] = wm[j] - 1e-5
        w_total = wu[j] + wl[j] + wd[j]
        if w_total > w_cap[j]:
            scale = w_cap[j] / w_total if w_total > 1e-10 else 1.0
            wu[j] *= scale
            wl[j] *= scale
            wd[j] *= scale

    for t in range(n_time):
        for j in range(n_basin):
            prcp = p_and_e[t, j, 0]
            if prcp < 0.0:
                prcp = 0.0
            pet = p_and_e[t, j, 1] * k[j]
            if pet < 0.0:
                pet = 0.0

            if wu[j] + prcp >= pet:
                eu = pet
            else:
                eu = wu[j] + prcp
            c_lm = c[j] * lm[j]
            pet_eu = pet - eu
            if wl[j] < c_lm and wl[j] < c[j] * pet_eu:
                ed = c[j] * pet_eu - wl[j]
            else:
                ed = 0.0
            if wu[j] + prcp >= pet:
                el = 0.0
            elif wl[j] >= c_lm:
                el = pet_eu * wl[j] / lm[j]
            elif wl[j] >= c[j] * pet_eu:
                el = c[j] * pet_eu
            else:
                el = wl[j]

            et = eu + el + ed
            prcp_diff = prcp - et
            pe = prcp_diff if prcp_diff > 0.0 else 0.0

            w0 = wu[j] + wl[j] + wd[j]
            if w0 < 0.0:
                w0 = 0.0
            if w0 > w_cap[j]:
                w0 = w_cap[j]
            wmm = wm[j] * (1.0 + b[j])
            ratio = 1.0 - w0 / wm[j]
            if ratio < 0.0:
                ratio = 0.0
            if ratio > 1.0:
                ratio = 1.0
            a = wmm * (1.0 - ratio ** (1.0 / (1.0 + b[j])))
            if np.isnan(a):
                a = 0.0

            r = 0.0
            rim = 0.0
            if pe > 0.0:
                if pe + a < wmm:
                    r = (
                        pe
                        - (wm[j] - w0)
                        + wm[j] * (1.0 - min(a + pe, wmm) / wmm) ** (1.0 + b[j])
                    )
                else:
                    r = pe - (wm[j] - w0)
                if r < 0.0:
                    r = 0.0
                rim = pe * im[j]
                if rim < 0.0:
                    rim = 0.0

            if prcp_diff > 0.0:
                wu_old = wu[j]
                wl_old = wl[j]
                wd_old = wd[j]
                wu_n = wu_old + prcp_diff - r
                if wu_n < um[j]:
                    wu_out = wu_n
                else:
                    wu_out = um[j]
                if wu_old + wl_old + prcp_diff - r > um[j] + lm[j]:
                    wd_out = wu_old + wl_old + wd_old + prcp_diff - r - um[j] - lm[j]
                else:
                    wd_out = wd_old
                wl_out = wu_old + wl_old + wd_old + prcp_diff - r - wu_out - wd_out
                wu[j] = wu_out
                wl[j] = wl_out
                wd[j] = wd_out
            else:
                wu_n = wu[j] + prcp_diff
                if wu_n > 0.0:
                    wu[j] = wu_n
                else:
                    wu[j] = 0.0
                wd[j] = wd[j] - ed
                wl[j] = wl[j] - el

            if wu[j] < 0.0:
                wu[j] = 0.0
            if wu[j] > um[j]:
                wu[j] = um[j]
            if wl[j] < 0.0:
                wl[j] = 0.0
            if wl[j] > lm[j]:
                wl[j] = lm[j]
            if wd[j] < 0.0:
                wd[j] = 0.0
            if wd[j] > dm[j]:
                wd[j] = dm[j]

            ms = sm[j] * (1.0 + ex[j])
            rs = 0.0
            ri = 0.0
            rg = 0.0
            if r > 0.0 and pe >= pe_eps:
                fr0 = fr[j]
                fr_j = r / pe
                if fr_j > fr_max:
                    fr_j = fr_max
                fr[j] = fr_j
                denom = fr_j if fr_j > 1e-10 else 1e-10
                ss = fr0 * s[j] / denom
                if ss > sm[j]:
                    ss = sm[j]
                au = ms * (1.0 - (1.0 - ss / sm[j]) ** (1.0 / (1.0 + ex[j])))
                if np.isnan(au):
                    au = 0.0
                if pe + au < ms:
                    rs = fr_j * (
                        pe
                        - sm[j]
                        + ss
                        + sm[j] * (1.0 - min(pe + au, ms) / ms) ** (1.0 + ex[j])
                    )
                else:
                    rs = fr_j * (pe + ss - sm[j])
                if rs > r:
                    rs = r
                s[j] = ss + (r - rs) / denom
                if s[j] > sm[j]:
                    s[j] = sm[j]
            ri = ki[j] * s[j] * fr[j]
            rg = kg[j] * s[j] * fr[j]
            s[j] = s[j] * (1.0 - ki[j] - kg[j])
            if s[j] < 0.0:
                s[j] = 0.0

            im_factor = 1.0 - im[j]
            runoff_im[t, j] = rim
            rss[t, j] = rs * im_factor
            ris[t, j] = ri * im_factor
            rgs[t, j] = rg * im_factor
            ets[t, j] = et

    return runoff_im, rss, ris, rgs, ets


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


@jit(nopython=True)
def route_csl(rss, ris, rgs, runoff_im, cs, l, ci, cg, qi0, qg0):
    n_time, n_basin = rss.shape
    qs = np.zeros((n_time, n_basin))
    qt = np.zeros((n_time, n_basin))
    qi = np.empty(n_basin)
    qg = np.empty(n_basin)
    for j in range(n_basin):
        qi[j] = qi0[j]
        qg[j] = qg0[j]
    for t in range(n_time):
        for j in range(n_basin):
            qi[j] = ci[j] * qi[j] + (1.0 - ci[j]) * ris[t, j]
            qg[j] = cg[j] * qg[j] + (1.0 - cg[j]) * rgs[t, j]
            qt[t, j] = rss[t, j] + runoff_im[t, j] + qi[j] + qg[j]
    for j in range(n_basin):
        lag = int(l[j])
        if lag < 0:
            lag = 0
        effective_lag = lag
        if effective_lag > n_time - 1:
            effective_lag = n_time - 1
        for t in range(effective_lag):
            qs[t, j] = qt[t, j]
        for t in range(effective_lag, n_time):
            qs[t, j] = cs[j] * qs[t - 1, j] + (1.0 - cs[j]) * qt[t - effective_lag, j]
    return qs


@jit(nopython=True)
def route_linear(ris, rgs, qs_surface, ci, cg, qi0, qg0):
    n_time, n_basin = ris.shape
    qs = np.zeros((n_time, n_basin))
    qi = np.empty(n_basin)
    qg = np.empty(n_basin)
    for j in range(n_basin):
        qi[j] = qi0[j]
        qg[j] = qg0[j]
    for t in range(n_time):
        for j in range(n_basin):
            qi[j] = ci[j] * qi[j] + (1.0 - ci[j]) * ris[t, j]
            qg[j] = cg[j] * qg[j] + (1.0 - cg[j]) * rgs[t, j]
            qs[t, j] = qs_surface[t, j] + qi[j] + qg[j]
    return qs


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------


def xaj(p_and_e, parameters, warmup_length=365):
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
    cs = params[:, 11]
    l = params[:, 12]
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
    qs = route_csl(rss, ris, rgs, runoff_im, cs, l, ci, cg, qi0, qg0)
    w = max(0, int(warmup_length))
    return qs[w:, :, np.newaxis], ets[w:, :]
