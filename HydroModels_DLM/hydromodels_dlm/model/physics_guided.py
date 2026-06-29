from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from hydromodels_dlm.config.model_config import MODEL_INPUT_KEYS, PBM_PARAM_BOUNDS

from hydromodels_dlm.model.LSTM import SeqRegLSTM
from hydromodels_dlm.model.dGR4J_CemaNeige import gthreshold_from_warmup
from hydromodels_dlm.model.dXAJ import initial_storage_from_forcing

# ---------------------------------------------------------------------------
# Model metadata
# ---------------------------------------------------------------------------


def is_physics_model(model_name):
    return str(model_name) in MODEL_INPUT_KEYS


def param_count(model_name):
    return len(PBM_PARAM_BOUNDS[model_name])


def param_names(model_name):
    return tuple(PBM_PARAM_BOUNDS[model_name].keys())


def driver_indices(model_name, input_keys):
    keys = MODEL_INPUT_KEYS[model_name]
    index = {k: i for i, k in enumerate(input_keys)}
    missing = [k for k in keys if k not in index]
    if missing:
        raise ValueError(
            f'model {model_name!r} needs dynamic_inputs {keys!r}; missing {missing!r}'
        )
    return [index[k] for k in keys]


def pbm_grad_steps(cfg, forecast_length):
    tc = cfg.get('training_cfgs') or {}
    if 'pbm_grad_steps' not in tc:
        return int(forecast_length)
    raw = tc['pbm_grad_steps']
    if raw is None:
        return int(forecast_length)
    return int(raw)

# ---------------------------------------------------------------------------
# Parameter mapping
# ---------------------------------------------------------------------------

def map_params_sigmoid(raw, low, high):
    low = torch.as_tensor(low, dtype=raw.dtype, device=raw.device)
    high = torch.as_tensor(high, dtype=raw.dtype, device=raw.device)
    return low + (high - low) * torch.sigmoid(raw)


def map_params_from_bounds(raw, bounds):
    names = tuple(bounds.keys())
    low = [bounds[n][0] for n in names]
    high = [bounds[n][1] for n in names]
    return map_params_sigmoid(raw, low, high), names

# ---------------------------------------------------------------------------
# Training forward
# ---------------------------------------------------------------------------


def predict_param_logits(param_predictor, lstm_inputs):
    gen = param_predictor(lstm_inputs)
    return gen[:, -1:, :]


def map_logits_to_physical_params(raw, pbm_name):
    bounds = PBM_PARAM_BOUNDS[pbm_name]
    return map_params_from_bounds(raw, bounds)


def predict_physical_params(param_predictor, lstm_inputs, pbm_name):
    gen = predict_param_logits(param_predictor, lstm_inputs)
    if not torch.isfinite(gen).all():
        raise ValueError('non-finite parameter logits')
    raw = gen[:, -1, :]
    return map_logits_to_physical_params(raw, pbm_name)


def physics_guided_forward(
    param_predictor,
    pb_core,
    *,
    drivers,
    lstm_inputs,
):
    warmup = int(pb_core.warmup_length)
    if drivers.shape[0] <= warmup:
        return drivers.new_full((0, drivers.shape[1], 1), float('nan'))

    raw = predict_param_logits(param_predictor, lstm_inputs)
    if not torch.isfinite(raw).all():
        return drivers.new_full(
            (drivers.shape[0] - warmup, drivers.shape[1], 1),
            float('nan'),
        )

    params, _ = map_logits_to_physical_params(raw[:, -1, :], pb_core.pbm_name)

    q = pb_core(drivers, params)
    if not torch.isfinite(q).all():
        return q.new_full((q.shape[0] - warmup, q.shape[1], 1), float('nan'))
    return q[warmup:]

# ---------------------------------------------------------------------------
# Numba runtime
# ---------------------------------------------------------------------------

NUMBA_OK = False
try:
    from numba import jit

    NUMBA_OK = True
except ImportError:
    jit = None

def require_numba():
    if not NUMBA_OK:
        raise RuntimeError('numba is required for sequential physics evaluate')

def numba_available():
    return NUMBA_OK


# ---------------------------------------------------------------------------
# Numpy adapters
# ---------------------------------------------------------------------------


def to_numpy_2d(x):
    if hasattr(x, 'detach'):
        arr = x.detach().cpu().double().numpy()
    else:
        arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def to_numpy_1d(x):
    return to_numpy_2d(x).reshape(-1)


def to_numpy_3d(x):
    if hasattr(x, 'detach'):
        return x.detach().cpu().double().numpy()
    return np.asarray(x, dtype=np.float64)


# ---------------------------------------------------------------------------
# Numba kernels
# ---------------------------------------------------------------------------

if NUMBA_OK:

    @jit(nopython=True)
    def frac_solid_usace(temp):
        tmin = -1.0
        tmax = 3.0
        if temp <= tmin:
            return 1.0
        if temp >= tmax:
            return 0.0
        return 1.0 - (temp - tmin) / (tmax - tmin)

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
        return perc + (pn - ps), s

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
    def route_store_step(q9, q1, x2, x3, r):
        if r < 0.0:
            r = 0.0
        if r > x3:
            r = x3
        exchange = x2 * (r / x3) ** 3.5
        r = r + q9 + exchange
        if r < 0.0:
            r = 0.0
        qr = r * (1.0 - (1.0 + (r / x3) ** 4) ** -0.25)
        r = r - qr
        qd = q1 + exchange
        if qd < 0.0:
            qd = 0.0
        return qr + qd, r

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
    def prefix_production(p, e, x1, s_init):
        n_time, n_basin = p.shape[0], p.shape[1]
        prs = np.zeros((n_time, n_basin), dtype=np.float64)
        s = s_init.copy()
        for t in range(n_time):
            for j in range(n_basin):
                prs[t, j], s[j] = production_step(p[t, j], e[t, j], x1[j], s[j])
        return prs, s

    @jit(nopython=True)
    def prefix_routing(q9, q1, x2, x3, r_init):
        n_time, n_basin = q9.shape[0], q9.shape[1]
        qsim = np.zeros((n_time, n_basin), dtype=np.float64)
        r = r_init.copy()
        for t in range(n_time):
            for j in range(n_basin):
                qsim[t, j], r[j] = route_store_step(
                    q9[t, j], q1[t, j], x2[j], x3[j], r[j]
                )
        return qsim, r

    @jit(nopython=True)
    def prefix_cemaneige(p, temp, cn1, cn2, gthreshold):
        n_time, n_basin = p.shape[0], p.shape[1]
        pliq = np.zeros((n_time, n_basin), dtype=np.float64)
        g = np.zeros(n_basin, dtype=np.float64)
        etg = np.zeros(n_basin, dtype=np.float64)
        for t in range(n_time):
            for j in range(n_basin):
                pliq[t, j], g[j], etg[j] = cemaneige_step(
                    p[t, j], temp[t, j], cn1[j], cn2[j], g[j], etg[j], gthreshold[j]
                )
        return pliq, g, etg

    @jit(nopython=True)
    def convolve_truncated(prs_col, w):
        n_time = prs_col.shape[0]
        m = w.shape[0]
        out = np.zeros(n_time, dtype=np.float64)
        for t in range(n_time):
            acc = 0.0
            for k in range(m):
                idx = t - k
                if idx >= 0:
                    acc += prs_col[idx] * w[k]
            out[t] = acc
        return out

    @jit(nopython=True)
    def xaj_runoff_series_numba(
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
                            + wm[j]
                            * (1.0 - min(a + pe, wmm) / wmm) ** (1.0 + b[j])
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
                        wd_out = (
                            wu_old + wl_old + wd_old + prcp_diff - r - um[j] - lm[j]
                        )
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
                if r > 0.0 and pe >= pe_eps:
                    fr0_j = fr[j]
                    fr_j = r / pe
                    if fr_j > fr_max:
                        fr_j = fr_max
                    fr[j] = fr_j
                    denom = fr_j if fr_j > 1e-10 else 1e-10
                    ss = fr0_j * s[j] / denom
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
                            + sm[j]
                            * (1.0 - min(pe + au, ms) / ms) ** (1.0 + ex[j])
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

        return runoff_im, rss, ris, rgs, wu, wl, wd, s, fr

    @jit(nopython=True)
    def constrain_ki_kg_pair(ki, kg):
        s = ki + kg
        if s > 0.99:
            scale = 0.99 / s
            return ki * scale, kg * scale
        return ki, kg

    @jit(nopython=True)
    def gamma_uh_weights(a_raw, theta_raw, len_uh):
        aa = a_raw + 0.1
        if aa < 0.0:
            aa = 0.0
        th = theta_raw + 0.5
        if th < 0.0:
            th = 0.0
        w = np.empty(len_uh, dtype=np.float64)
        denom = math.exp(math.lgamma(aa)) * (th ** aa)
        for i in range(len_uh):
            t = 0.5 + float(i)
            w[i] = (t ** (aa - 1.0)) * math.exp(-t / th) / denom
        total = 0.0
        for i in range(len_uh):
            total += w[i]
        if total > 0.0:
            for i in range(len_uh):
                w[i] /= total
        return w

    @jit(nopython=True)
    def scan_gr4j_production_varying(p, e, params_seq, s_init):
        n_time, n_basin = p.shape
        prs = np.zeros((n_time, n_basin), dtype=np.float64)
        s = s_init.copy()
        for t in range(n_time):
            for j in range(n_basin):
                prs[t, j], s[j] = production_step(
                    p[t, j], e[t, j], params_seq[t, j, 0], s[j]
                )
        return prs

    @jit(nopython=True)
    def convolve_gr4j_varying(prs, params_seq, quickflow_90):
        n_time, n_basin = prs.shape
        out = np.zeros((n_time, n_basin), dtype=np.float64)
        for t in range(n_time):
            eff_limit = t + 1
            for j in range(n_basin):
                w = unit_hydrograph_weights(params_seq[t, j, 3], quickflow_90)
                m = w.shape[0]
                eff = m if eff_limit >= m else eff_limit
                acc = 0.0
                for k in range(eff):
                    acc += w[k] * prs[t - k, j]
                out[t, j] = acc
        return out

    @jit(nopython=True)
    def route_gr4j_varying(q9, q1, params_seq, r_init):
        n_time, n_basin = q9.shape
        qsim = np.zeros((n_time, n_basin), dtype=np.float64)
        r = r_init.copy()
        for t in range(n_time):
            for j in range(n_basin):
                qsim[t, j], r[j] = route_store_step(
                    q9[t, j],
                    q1[t, j],
                    params_seq[t, j, 1],
                    params_seq[t, j, 2],
                    r[j],
                )
        return qsim

    @jit(nopython=True)
    def scan_cemaneige_varying(p, temp, params_seq, cn_offset, gthreshold):
        n_time, n_basin = p.shape
        pliq = np.zeros((n_time, n_basin), dtype=np.float64)
        g = np.zeros(n_basin, dtype=np.float64)
        etg = np.zeros(n_basin, dtype=np.float64)
        for t in range(n_time):
            for j in range(n_basin):
                pliq[t, j], g[j], etg[j] = cemaneige_step(
                    p[t, j],
                    temp[t, j],
                    params_seq[t, j, cn_offset],
                    params_seq[t, j, cn_offset + 1],
                    g[j],
                    etg[j],
                    gthreshold[j],
                )
        return pliq

    @jit(nopython=True)
    def route_xaj_qt_varying(runoff_im, rss, ris, rgs, params_seq, qi0, qg0):
        n_time, n_basin = ris.shape
        qi = qi0.copy()
        qg = qg0.copy()
        qt = np.zeros((n_time, n_basin), dtype=np.float64)
        for t in range(n_time):
            for j in range(n_basin):
                ci = params_seq[t, j, 13]
                cg = params_seq[t, j, 14]
                qi[j] = ci * qi[j] + (1.0 - ci) * ris[t, j]
                qg[j] = cg * qg[j] + (1.0 - cg) * rgs[t, j]
                qt[t, j] = rss[t, j] + runoff_im[t, j] + qi[j] + qg[j]
        return qt

    @jit(nopython=True)
    def route_dxaj_csl_varying(qt, params_seq):
        n_time, n_basin = qt.shape
        qsim = np.zeros((n_time, n_basin), dtype=np.float64)
        for j in range(n_basin):
            for t in range(n_time):
                cs = params_seq[t, j, 11]
                lag = int(round(params_seq[t, j, 12]))
                if lag < 0:
                    lag = 0
                effective_lag = lag if lag <= n_time - 1 else n_time - 1
                if t < effective_lag:
                    qsim[t, j] = qt[t, j]
                else:
                    qsim[t, j] = (
                        cs * qsim[t - 1, j]
                        + (1.0 - cs) * qt[t - effective_lag, j]
                    )
        return qsim

    @jit(nopython=True)
    def scan_xaj_runoff_varying(
        p_and_e,
        params_seq,
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
        wu = wu0.copy()
        wl = wl0.copy()
        wd = wd0.copy()
        s = s0.copy()
        fr = fr0.copy()

        for t in range(n_time):
            for j in range(n_basin):
                k = params_seq[t, j, 0]
                b = params_seq[t, j, 1]
                im = params_seq[t, j, 2]
                um = params_seq[t, j, 3]
                lm = params_seq[t, j, 4]
                dm = params_seq[t, j, 5]
                c = params_seq[t, j, 6]
                sm = params_seq[t, j, 7]
                ex = params_seq[t, j, 8]
                ki, kg = constrain_ki_kg_pair(
                    params_seq[t, j, 9],
                    params_seq[t, j, 10],
                )

                wm = um + lm + dm
                if wm < 1e-5:
                    wm = 1e-5
                w_cap = wm - 1e-5
                w_total = wu[j] + wl[j] + wd[j]
                if w_total > w_cap:
                    scale = w_cap / w_total if w_total > 1e-10 else 1.0
                    wu[j] *= scale
                    wl[j] *= scale
                    wd[j] *= scale

                prcp = p_and_e[t, j, 0]
                if prcp < 0.0:
                    prcp = 0.0
                pet = p_and_e[t, j, 1] * k
                if pet < 0.0:
                    pet = 0.0

                if wu[j] + prcp >= pet:
                    eu = pet
                else:
                    eu = wu[j] + prcp
                c_lm = c * lm
                pet_eu = pet - eu
                if wl[j] < c_lm and wl[j] < c * pet_eu:
                    ed = c * pet_eu - wl[j]
                else:
                    ed = 0.0
                if wu[j] + prcp >= pet:
                    el = 0.0
                elif wl[j] >= c_lm:
                    el = pet_eu * wl[j] / lm
                elif wl[j] >= c * pet_eu:
                    el = c * pet_eu
                else:
                    el = wl[j]

                et = eu + el + ed
                prcp_diff = prcp - et
                pe = prcp_diff if prcp_diff > 0.0 else 0.0

                w0 = wu[j] + wl[j] + wd[j]
                if w0 < 0.0:
                    w0 = 0.0
                if w0 > w_cap:
                    w0 = w_cap
                wmm = wm * (1.0 + b)
                ratio = 1.0 - w0 / wm
                if ratio < 0.0:
                    ratio = 0.0
                if ratio > 1.0:
                    ratio = 1.0
                a_store = wmm * (1.0 - ratio ** (1.0 / (1.0 + b)))
                if np.isnan(a_store):
                    a_store = 0.0

                r = 0.0
                rim = 0.0
                if pe > 0.0:
                    if pe + a_store < wmm:
                        r = (
                            pe
                            - (wm - w0)
                            + wm
                            * (1.0 - min(pe + a_store, wmm) / wmm) ** (1.0 + b)
                        )
                    else:
                        r = pe - (wm - w0)
                    if r < 0.0:
                        r = 0.0
                    rim = pe * im
                    if rim < 0.0:
                        rim = 0.0

                if prcp_diff > 0.0:
                    wu_old = wu[j]
                    wl_old = wl[j]
                    wd_old = wd[j]
                    wu_n = wu_old + prcp_diff - r
                    if wu_n < um:
                        wu_out = wu_n
                    else:
                        wu_out = um
                    if wu_old + wl_old + prcp_diff - r > um + lm:
                        wd_out = (
                            wu_old + wl_old + wd_old + prcp_diff - r - um - lm
                        )
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
                if wu[j] > um:
                    wu[j] = um
                if wl[j] < 0.0:
                    wl[j] = 0.0
                if wl[j] > lm:
                    wl[j] = lm
                if wd[j] < 0.0:
                    wd[j] = 0.0
                if wd[j] > dm:
                    wd[j] = dm

                ms = sm * (1.0 + ex)
                rs = 0.0
                if r > 0.0 and pe >= pe_eps:
                    fr0_j = fr[j]
                    fr_j = r / pe
                    if fr_j > fr_max:
                        fr_j = fr_max
                    fr[j] = fr_j
                    denom = fr_j if fr_j > 1e-10 else 1e-10
                    ss = fr0_j * s[j] / denom
                    if ss > sm:
                        ss = sm
                    au = ms * (1.0 - (1.0 - ss / sm) ** (1.0 / (1.0 + ex)))
                    if np.isnan(au):
                        au = 0.0
                    if pe + au < ms:
                        rs = fr_j * (
                            pe
                            - sm
                            + ss
                            + sm
                            * (1.0 - min(pe + au, ms) / ms) ** (1.0 + ex)
                        )
                    else:
                        rs = fr_j * (pe + ss - sm)
                    if rs > r:
                        rs = r
                    s[j] = ss + (r - rs) / denom
                    if s[j] > sm:
                        s[j] = sm
                ri = ki * s[j] * fr[j]
                rg = kg * s[j] * fr[j]
                s[j] = s[j] * (1.0 - ki - kg)
                if s[j] < 0.0:
                    s[j] = 0.0

                im_factor = 1.0 - im
                runoff_im[t, j] = rim
                rss[t, j] = rs * im_factor
                ris[t, j] = ri * im_factor
                rgs[t, j] = rg * im_factor

        return runoff_im, rss, ris, rgs, wu, wl, wd, s, fr

    @jit(nopython=True)
    def route_dxaj_mz_varying(
        runoff_im,
        rss,
        ris,
        rgs,
        params_seq,
        qi0,
        qg0,
        kernel_size,
    ):
        n_time, n_basin = ris.shape
        qi = qi0.copy()
        qg = qg0.copy()
        qsim = np.zeros((n_time, n_basin), dtype=np.float64)
        for t in range(n_time):
            eff = kernel_size if t + 1 >= kernel_size else t + 1
            for j in range(n_basin):
                a_raw = params_seq[t, j, 11]
                theta_raw = params_seq[t, j, 12]
                ci = params_seq[t, j, 13]
                cg = params_seq[t, j, 14]
                w = gamma_uh_weights(a_raw, theta_raw, eff)
                acc = 0.0
                for k in range(eff):
                    acc += w[k] * (runoff_im[t - k, j] + rss[t - k, j])
                qi[j] = ci * qi[j] + (1.0 - ci) * ris[t, j]
                qg[j] = cg * qg[j] + (1.0 - cg) * rgs[t, j]
                qsim[t, j] = acc + qi[j] + qg[j]
        return qsim


# ---------------------------------------------------------------------------
# Prefix adapters
# ---------------------------------------------------------------------------

def prefix_gr4j_production(p, e, x1, *, s_init=None):
    require_numba()
    p_np = to_numpy_2d(p)
    e_np = to_numpy_2d(e)
    x1_np = to_numpy_1d(x1)
    if s_init is None:
        s_init = 0.3 * x1_np
    else:
        s_init = to_numpy_1d(s_init)
    return prefix_production(p_np, e_np, x1_np, s_init)


def prefix_gr4j_routing(prs, x2, x3, x4, *, r_init=None):
    require_numba()
    prs_np = to_numpy_2d(prs)
    x2_np = to_numpy_1d(x2)
    x3_np = to_numpy_1d(x3)
    x4_np = to_numpy_1d(x4)
    n_basin = prs_np.shape[1]
    q9 = np.zeros_like(prs_np)
    q1 = np.zeros_like(prs_np)
    for j in range(n_basin):
        w90 = unit_hydrograph_weights(x4_np[j], True)
        w10 = unit_hydrograph_weights(x4_np[j], False)
        q9[:, j] = convolve_truncated(prs_np[:, j], w90)
        q1[:, j] = convolve_truncated(prs_np[:, j], w10)
    if r_init is None:
        r_init = 0.5 * x3_np
    else:
        r_init = to_numpy_1d(r_init)
    return prefix_routing(q9, q1, x2_np, x3_np, r_init)


def prefix_cemaneige_states(p, temp, cn1, cn2, gthreshold):
    require_numba()
    return prefix_cemaneige(
        to_numpy_2d(p),
        to_numpy_2d(temp),
        to_numpy_1d(cn1),
        to_numpy_1d(cn2),
        to_numpy_1d(gthreshold),
    )


def constrain_ki_kg_numpy(params):
    arr = np.asarray(params, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim == 3:
        s = arr[:, :, 9] + arr[:, :, 10]
        over = s > 0.99
        if np.any(over):
            scale = np.where(over, 0.99 / s, 1.0)
            arr = arr.copy()
            arr[:, :, 9] = arr[:, :, 9] * scale
            arr[:, :, 10] = arr[:, :, 10] * scale
        return arr
    s = arr[:, 9] + arr[:, 10]
    over = s > 0.99
    if np.any(over):
        scale = np.where(over, 0.99 / s, 1.0)
        arr = arr.copy()
        arr[:, 9] = arr[:, 9] * scale
        arr[:, 10] = arr[:, 10] * scale
    return arr


def prefix_xaj_runoff(p_and_e, params, wu0, wl0, wd0, s0, fr0):
    require_numba()
    ndim = p_and_e.dim() if hasattr(p_and_e, 'dim') else np.asarray(p_and_e).ndim
    pe_np = to_numpy_3d(p_and_e) if ndim == 3 else to_numpy_2d(p_and_e)
    if pe_np.ndim == 2:
        pe_np = pe_np.reshape(pe_np.shape[0], pe_np.shape[1], 1)
    params_np = constrain_ki_kg_numpy(to_numpy_2d(params))
    return xaj_runoff_series_numba(
        pe_np,
        params_np[:, 0],
        params_np[:, 1],
        params_np[:, 2],
        params_np[:, 3],
        params_np[:, 4],
        params_np[:, 5],
        params_np[:, 6],
        params_np[:, 7],
        params_np[:, 8],
        params_np[:, 9],
        params_np[:, 10],
        to_numpy_1d(wu0),
        to_numpy_1d(wl0),
        to_numpy_1d(wd0),
        to_numpy_1d(s0),
        to_numpy_1d(fr0),
    )

# ---------------------------------------------------------------------------
# Sequential inference (evaluate)
# ---------------------------------------------------------------------------


def predict_param_trajectory(param_predictor, lstm_inputs, pbm_name):
    batch = lstm_inputs.unsqueeze(0) if lstm_inputs.dim() == 2 else lstm_inputs
    gen = param_predictor(batch)
    params, _ = map_logits_to_physical_params(gen[0], pbm_name)
    return params


def align_params_seq_length(params_seq, n_time):
    if params_seq.shape[0] != n_time:
        raise ValueError(
            f'params_seq length {params_seq.shape[0]} != drivers length {n_time}'
        )


def xaj_initial_storage(p_and_e, params_seq, warmup_length):
    n_time = p_and_e.shape[0]
    n_basin = p_and_e.shape[1]
    prefix_len = max(1, min(int(warmup_length), n_time))
    prefix = params_seq[:prefix_len]
    wu0, wl0, wd0, s0 = initial_storage_from_forcing(
        p_and_e,
        prefix[:, :, 3].mean(dim=0),
        prefix[:, :, 4].mean(dim=0),
        prefix[:, :, 5].mean(dim=0),
        prefix[:, :, 7].mean(dim=0),
        warmup_length=prefix_len,
    )
    fr0 = p_and_e.new_full((n_basin,), 0.1)
    qi0 = p_and_e.new_full((n_basin,), 0.1)
    qg0 = p_and_e.new_full((n_basin,), 0.1)
    return wu0, wl0, wd0, s0, fr0, qi0, qg0


def resolve_gthreshold(pb_core, p, temp, warmup_length):
    gbuf = getattr(pb_core, 'gthreshold_const', None)
    if gbuf is not None and gbuf.numel() > 0:
        gt = gbuf.to(device=p.device, dtype=p.dtype).reshape(-1)
        if gt.shape[0] == 1 and p.shape[1] > 1:
            gt = gt.expand(p.shape[1])
        return gt
    return gthreshold_from_warmup(p, temp, warmup_length)



def qsim_tensor_from_numpy(q_np, ref):
    out = torch.from_numpy(np.asarray(q_np, dtype=np.float64)).to(
        device=ref.device,
        dtype=ref.dtype,
    )
    if out.ndim == 1:
        return out.unsqueeze(-1)
    return out


def build_precip_pet_array(pliq, pet):
    pe_np = np.empty((pliq.shape[0], pliq.shape[1], 2), dtype=np.float64)
    pe_np[:, :, 0] = pliq
    pe_np[:, :, 1] = pet
    return pe_np


def simulate_dgr4j_varying(pe_np, params_np):
    require_numba()
    p_np = pe_np[:, :, 0]
    e_np = pe_np[:, :, 1]
    s0 = 0.3 * params_np[0, :, 0]
    prs = scan_gr4j_production_varying(p_np, e_np, params_np, s0)
    q9 = convolve_gr4j_varying(prs, params_np, True)
    q1 = convolve_gr4j_varying(prs, params_np, False)
    r0 = 0.5 * params_np[0, :, 2]
    return route_gr4j_varying(q9, q1, params_np, r0)


def simulate_xaj_varying(pe_np, params_np, *, warmup, kernel_size, pbm_name, ref):
    require_numba()
    params = torch.from_numpy(params_np).to(device=ref.device, dtype=ref.dtype)
    drivers = torch.from_numpy(pe_np).to(device=ref.device, dtype=ref.dtype)
    wu0, wl0, wd0, s0, fr0, qi0, qg0 = xaj_initial_storage(drivers, params, warmup)
    runoff_im, rss, ris, rgs, _, _, _, _, _ = scan_xaj_runoff_varying(
        pe_np,
        params_np,
        to_numpy_1d(wu0),
        to_numpy_1d(wl0),
        to_numpy_1d(wd0),
        to_numpy_1d(s0),
        to_numpy_1d(fr0),
    )
    qi0_np = to_numpy_1d(qi0)
    qg0_np = to_numpy_1d(qg0)
    if pbm_name == 'dXAJ-mz':
        return route_dxaj_mz_varying(
            runoff_im,
            rss,
            ris,
            rgs,
            params_np,
            qi0_np,
            qg0_np,
            int(kernel_size),
        )
    qt = route_xaj_qt_varying(
        runoff_im, rss, ris, rgs, params_np, qi0_np, qg0_np
    )
    return route_dxaj_csl_varying(qt, params_np)


def run_cemaneige_varying(drivers, params_np, cn_offset, gthreshold):
    require_numba()
    return scan_cemaneige_varying(
        to_numpy_2d(drivers[:, :, 0]),
        to_numpy_2d(drivers[:, :, 2]),
        params_np,
        cn_offset,
        gthreshold,
    )


def simulate_physics_varying(pb_core, drivers_tbn, params_tbn, *, warmup, kernel_size, ref):
    require_numba()
    name = pb_core.pbm_name
    if not is_physics_model(name):
        raise ValueError(f'{name!r} is not a registered physics model')

    drivers_tbn = np.asarray(drivers_tbn, dtype=np.float64)
    params_tbn = np.asarray(params_tbn, dtype=np.float64)
    if drivers_tbn.ndim != 3 or params_tbn.ndim != 3:
        raise ValueError(
            f'drivers/params must be [T, B, ·]; got '
            f'{drivers_tbn.shape} and {params_tbn.shape}'
        )
    if drivers_tbn.shape[0] != params_tbn.shape[0]:
        raise ValueError(
            f'drivers length {drivers_tbn.shape[0]} != params length {params_tbn.shape[0]}'
        )

    if name == 'dGR4J':
        return simulate_dgr4j_varying(drivers_tbn[:, :, :2], params_tbn)

    if name in ('dXAJ', 'dXAJ-mz'):
        return simulate_xaj_varying(
            drivers_tbn[:, :, :2],
            constrain_ki_kg_numpy(params_tbn),
            warmup=warmup,
            kernel_size=kernel_size,
            pbm_name=name,
            ref=ref,
        )

    drivers_dev = torch.from_numpy(drivers_tbn).to(device=ref.device, dtype=ref.dtype)
    gthreshold = to_numpy_1d(
        resolve_gthreshold(pb_core, drivers_dev[:, :, 0], drivers_dev[:, :, 2], warmup)
    )
    pet_np = drivers_tbn[:, :, 1]

    if name == 'dGR4J-CemaNeige':
        pliq = run_cemaneige_varying(drivers_tbn, params_tbn, 4, gthreshold)
        pe_np = build_precip_pet_array(pliq, pet_np)
        return simulate_dgr4j_varying(pe_np, params_tbn[:, :, :4])

    if name == 'dXAJ-mz-CemaNeige':
        pliq = run_cemaneige_varying(drivers_tbn, params_tbn, 15, gthreshold)
        pe_np = build_precip_pet_array(pliq, pet_np)
        return simulate_xaj_varying(
            pe_np,
            constrain_ki_kg_numpy(params_tbn[:, :, :15]),
            warmup=warmup,
            kernel_size=kernel_size,
            pbm_name='dXAJ-mz',
            ref=ref,
        )

    raise ValueError(f'no numba eval kernel for physics model {name!r}')


def run_sequential_physics_sim(pb_core, drivers, params_seq, *, warmup, kernel_size):
    require_numba()
    align_params_seq_length(params_seq, drivers.shape[0])
    qsim = simulate_physics_varying(
        pb_core,
        to_numpy_3d(drivers),
        to_numpy_3d(params_seq.to(dtype=drivers.dtype)),
        warmup=warmup,
        kernel_size=kernel_size,
        ref=drivers,
    )
    return qsim_tensor_from_numpy(qsim, drivers)


def simulate_sliding_physics(
    pb_core,
    drivers_windows,
    params_batch,
    *,
    warmup,
    forecast,
    kernel_size,
    ref,
):
    drivers_btd = np.asarray(drivers_windows, dtype=np.float64)
    if drivers_btd.ndim != 3:
        raise ValueError(
            f'drivers_windows must be [B, T, D], got shape {drivers_btd.shape}'
        )
    batch_size, n_time, _ = drivers_btd.shape
    drivers_tbn = np.transpose(drivers_btd, (1, 0, 2))

    params = np.asarray(params_batch, dtype=np.float64)
    if params.ndim == 1:
        params = params.reshape(1, -1)
    params_tbn = np.broadcast_to(
        params[np.newaxis, :, :],
        (n_time, batch_size, params.shape[1]),
    ).copy()

    qsim = simulate_physics_varying(
        pb_core,
        drivers_tbn,
        params_tbn,
        warmup=warmup,
        kernel_size=kernel_size,
        ref=ref,
    )
    out_idx = int(warmup) + int(forecast) - 1
    if out_idx < 0 or out_idx >= n_time:
        raise ValueError(
            f'forecast step {out_idx} out of window length {n_time}'
        )
    return np.asarray(qsim[out_idx, :], dtype=np.float64)


def sequential_physics_qsim(
    model,
    drivers,
    lstm_seq,
    device,
    *,
    warmup,
    kernel_size=15,
    eval_idx=None,
):
    drivers_np = (
        drivers.detach().cpu().numpy()
        if isinstance(drivers, torch.Tensor)
        else np.asarray(drivers)
    )
    lstm_np = (
        lstm_seq.detach().cpu().numpy()
        if isinstance(lstm_seq, torch.Tensor)
        else np.asarray(lstm_seq)
    )
    if drivers_np.ndim == 2:
        drivers_np = drivers_np.reshape(-1, 1, drivers_np.shape[-1])

    model.eval()
    with torch.inference_mode():
        drivers_dev = torch.from_numpy(drivers_np.astype(np.float32)).to(device)
        lstm_dev = torch.from_numpy(lstm_np.astype(np.float32)).to(device)
        params_seq = predict_param_trajectory(
            model.param_predictor,
            lstm_dev,
            model.pbm_name,
        ).unsqueeze(1)
        q = run_sequential_physics_sim(
            model.pb_core,
            drivers_dev,
            params_seq,
            warmup=int(warmup),
            kernel_size=int(kernel_size),
        )
        qsim = q[int(warmup) :, 0].detach().cpu().numpy().astype(np.float64)

    if eval_idx is not None:
        return qsim[np.asarray(eval_idx, dtype=np.int64)]
    return qsim


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------


class PhysicsGuidedModel(nn.Module):

    def __init__(self, param_predictor, pb_core):
        super().__init__()
        self.param_predictor = param_predictor
        self.pb_core = pb_core
        self.pbm_name = pb_core.pbm_name

    @property
    def warmup_length(self):
        return int(self.pb_core.warmup_length)

    @property
    def n_params(self):
        return param_count(self.pbm_name)

    def forward(self, drivers, lstm_inputs):
        return physics_guided_forward(
            self.param_predictor,
            self.pb_core,
            drivers=drivers,
            lstm_inputs=lstm_inputs,
        )


def make_physics_model(
    *,
    n_lstm_inputs,
    output_size,
    pb_core,
    hidden_size=32,
    dropout=0.0,
    input_proj=None,
):
    predictor = SeqRegLSTM(
        input_size=int(n_lstm_inputs),
        output_size=int(output_size),
        hidden_size=hidden_size,
        dropout=dropout,
        input_proj=input_proj,
    )
    return PhysicsGuidedModel(predictor, pb_core)


def apply_pbm_grad_steps(model, cfg, forecast_length):
    if not isinstance(model, PhysicsGuidedModel):
        return
    model.pb_core.pbm_grad_steps = pbm_grad_steps(cfg, forecast_length)


def apply_cemaneige_climatology(model, loader, basin_ids):
    pb_core = getattr(model, 'pb_core', None)
    if pb_core is None or not hasattr(pb_core, 'set_gthreshold_climatology'):
        return

    driver_idx = driver_indices(model.pbm_name, loader.input_keys)
    bid = str(basin_ids[0])
    x_raw, _, _ = loader.arrays_for_period(bid, 'train')
    pb_core.set_gthreshold_climatology(
        x_raw[:, driver_idx[0]],
        x_raw[:, driver_idx[2]],
        loader.warmup_length,
    )

