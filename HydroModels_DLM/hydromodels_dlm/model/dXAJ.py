"""Differentiable XAJ (fused runoff + CSL routing), aligned with HydroModels_PBM XAJ."""

import torch
from torch import nn

from hydromodels_dlm.model.dGR4J import effective_grad_steps


# ---------------------------------------------------------------------------
# Parameter constraint
# ---------------------------------------------------------------------------


def constrain_ki_kg(params, *, ki_idx=9, kg_idx=10, max_sum=0.99):
    ki = params[:, ki_idx]
    kg = params[:, kg_idx]
    s = ki + kg
    over = s > max_sum
    scale = torch.where(over, max_sum / s, torch.ones_like(s))
    out = params.clone()
    out[:, ki_idx] = ki * scale
    out[:, kg_idx] = kg * scale
    return out


# ---------------------------------------------------------------------------
# Initial storage
# ---------------------------------------------------------------------------


def initial_storage_from_forcing(p_and_e, um, lm, dm, sm, warmup_length=365):
    prcp = torch.clamp(p_and_e[:, :, 0], min=0.0)
    pet = torch.clamp(p_and_e[:, :, 1], min=0.0)
    n_time = prcp.shape[0]
    n = max(1, min(int(warmup_length), n_time))
    p_bar = prcp[:n].mean(dim=0)
    e_bar = pet[:n].mean(dim=0)
    moisture = torch.clamp(p_bar / torch.clamp(e_bar, min=1e-6), min=0.5, max=1.5)
    wu = torch.clamp(0.5 * um * moisture, min=0.15 * um, max=0.85 * um)
    wl = torch.clamp(0.5 * lm * moisture, min=0.15 * lm, max=0.85 * lm)
    wd = 0.5 * dm
    s = torch.clamp(0.5 * sm * moisture, min=0.15 * sm, max=0.85 * sm)
    return wu, wl, wd, s


# ---------------------------------------------------------------------------
# Runoff (fused production + source separation, PBM-aligned)
# ---------------------------------------------------------------------------

# Mirrors HydroModels_PBM/hydromodels/model/XAJ.py::calc_runoff_series.
# Extra clamps below stabilise autograd; Numba prefix keeps PBM formulas.


def clamped_pow(base, exponent, *, lo=1e-8, hi=1.0):
    return torch.clamp(base, min=lo, max=hi) ** exponent


def xaj_runoff_step(
    prcp,
    pet,
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
    wu,
    wl,
    wd,
    s,
    fr,
    wm,
    w_cap,
):
    pe_eps = 1e-6
    fr_max = 1.0

    pet = torch.clamp(pet * k, min=0.0)
    prcp = torch.clamp(prcp, min=0.0)

    eu = torch.where(wu + prcp >= pet, pet, wu + prcp)
    c_lm = c * lm
    pet_eu = pet - eu
    ed = torch.where(
        (wl < c_lm) & (wl < c * pet_eu),
        c * pet_eu - wl,
        torch.zeros_like(wl),
    )
    lm_safe = torch.clamp(lm, min=1e-4)
    el = torch.where(
        wu + prcp >= pet,
        torch.zeros_like(wl),
        torch.where(
            wl >= c_lm,
            pet_eu * wl / lm_safe,
            torch.where(wl >= c * pet_eu, c * pet_eu, wl),
        ),
    )
    et = eu + el + ed
    prcp_diff = prcp - et
    pe = torch.clamp(prcp_diff, min=0.0)

    w0 = torch.clamp(wu + wl + wd, min=0.0)
    w0 = torch.minimum(w0, w_cap)
    wmm = wm * (1.0 + b)
    ratio = torch.clamp(1.0 - w0 / wm, min=0.0, max=1.0)
    a = wmm * (1.0 - clamped_pow(ratio, 1.0 / (1.0 + b)))
    a = torch.where(torch.isnan(a), torch.zeros_like(a), a)

    r = torch.zeros_like(pe)
    rim = torch.zeros_like(pe)
    has_pe = pe > 0.0
    small = has_pe & (pe + a < wmm)
    large = has_pe & ~small
    wmm_ratio = torch.clamp(a + pe, max=wmm) / torch.clamp(wmm, min=1e-8)
    r = torch.where(
        small,
        pe - (wm - w0) + wm * clamped_pow(1.0 - wmm_ratio, 1.0 + b),
        r,
    )
    r = torch.where(large, pe - (wm - w0), r)
    r = torch.clamp(r, min=0.0)
    rim = torch.clamp(pe * im, min=0.0)

    pos_diff = prcp_diff > 0.0
    wu_old, wl_old, wd_old = wu, wl, wd
    wu_n = wu_old + prcp_diff - r
    wu_out = torch.minimum(wu_n, um)
    total_after = wu_old + wl_old + prcp_diff - r
    wd_out = torch.where(
        total_after > um + lm,
        wu_old + wl_old + wd_old + prcp_diff - r - um - lm,
        wd_old,
    )
    wl_out = wu_old + wl_old + wd_old + prcp_diff - r - wu_out - wd_out
    wu = torch.where(pos_diff, wu_out, torch.clamp(wu + prcp_diff, min=0.0))
    wl = torch.where(pos_diff, wl_out, wl - el)
    wd = torch.where(pos_diff, wd_out, wd - ed)

    wu = torch.clamp(wu, min=0.0)
    wu = torch.minimum(wu, um)
    wl = torch.clamp(wl, min=0.0)
    wl = torch.minimum(wl, lm)
    wd = torch.clamp(wd, min=0.0)
    wd = torch.minimum(wd, dm)

    ms = sm * (1.0 + ex)
    rs = torch.zeros_like(r)
    fr0 = fr
    has_runoff = (r > 0.0) & (pe >= pe_eps)
    fr_j = torch.where(
        has_runoff,
        torch.clamp(r / torch.clamp(pe, min=pe_eps), max=fr_max),
        fr,
    )
    denom = torch.clamp(fr_j, min=1e-10)
    sm_safe = torch.clamp(sm, min=1e-4)
    ss = torch.where(has_runoff, torch.clamp(fr0 * s / denom, max=sm_safe), s)
    ss_ratio = torch.clamp(ss / sm_safe, min=0.0, max=1.0)
    au = ms * (1.0 - clamped_pow(1.0 - ss_ratio, 1.0 / (1.0 + ex)))
    au = torch.where(torch.isnan(au), torch.zeros_like(au), au)
    ms_ratio = torch.clamp(pe + au, max=ms) / torch.clamp(ms, min=1e-8)
    rs_small = fr_j * (pe - sm_safe + ss + sm_safe * clamped_pow(1.0 - ms_ratio, 1.0 + ex))
    rs_large = fr_j * (pe + ss - sm)
    rs = torch.where(has_runoff, torch.where(pe + au < ms, rs_small, rs_large), rs)
    rs = torch.minimum(rs, r)
    s = torch.where(
        has_runoff,
        torch.clamp(ss + (r - rs) / denom, max=sm_safe),
        s,
    )
    fr = torch.where(has_runoff, fr_j, fr)

    ri = ki * s * fr
    rg = kg * s * fr
    s = torch.clamp(s * (1.0 - ki - kg), min=0.0)

    im_factor = 1.0 - im
    return (
        rim,
        rs * im_factor,
        ri * im_factor,
        rg * im_factor,
        et,
        wu,
        wl,
        wd,
        s,
        fr,
    )


def prepare_runoff_storage_caps(um, lm, dm, wu, wl, wd):
    wm = um + lm + dm
    wm = torch.clamp(wm, min=1e-5)
    w_cap = wm - 1e-5
    w_total = wu + wl + wd
    over = w_total > w_cap
    scale = torch.where(over, w_cap / torch.clamp(w_total, min=1e-10), torch.ones_like(w_total))
    wu = wu * scale
    wl = wl * scale
    wd = wd * scale
    return wu, wl, wd, wm, w_cap


def scan_runoff_series(
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
    n_time = p_and_e.shape[0]
    n_basin = p_and_e.shape[1]
    runoff_im = p_and_e.new_zeros(n_time, n_basin)
    rss = p_and_e.new_zeros(n_time, n_basin)
    ris = p_and_e.new_zeros(n_time, n_basin)
    rgs = p_and_e.new_zeros(n_time, n_basin)

    wu, wl, wd, wm, w_cap = prepare_runoff_storage_caps(um, lm, dm, wu0, wl0, wd0)
    s = s0
    fr = fr0

    for t in range(n_time):
        prcp = p_and_e[t, :, 0]
        pet = p_and_e[t, :, 1]
        rim, rs, ri, rg, _, wu, wl, wd, s, fr = xaj_runoff_step(
            prcp,
            pet,
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
            wu,
            wl,
            wd,
            s,
            fr,
            wm,
            w_cap,
        )
        runoff_im[t] = rim
        rss[t] = rs
        ris[t] = ri
        rgs[t] = rg

    return runoff_im, rss, ris, rgs, wu, wl, wd, s, fr


def prefix_runoff_no_grad(p_and_e, params, wu0, wl0, wd0, s0, fr0):
    with torch.no_grad():
        return scan_runoff_from_params(p_and_e, params, wu0, wl0, wd0, s0, fr0)


def prefix_runoff_numba(p_and_e, params, wu0, wl0, wd0, s0, fr0, device, dtype):
    from hydromodels_dlm.model.physics_guided import numba_available, prefix_xaj_runoff

    if not numba_available():
        return prefix_runoff_no_grad(p_and_e, params, wu0, wl0, wd0, s0, fr0)
    arrays = prefix_xaj_runoff(p_and_e, params, wu0, wl0, wd0, s0, fr0)
    return tuple(torch.as_tensor(arr, device=device, dtype=dtype) for arr in arrays)


def scan_runoff_from_params(p_and_e, params, wu0, wl0, wd0, s0, fr0):
    params = constrain_ki_kg(params)
    return scan_runoff_series(
        p_and_e,
        params[:, 0],
        params[:, 1],
        params[:, 2],
        params[:, 3],
        params[:, 4],
        params[:, 5],
        params[:, 6],
        params[:, 7],
        params[:, 8],
        params[:, 9],
        params[:, 10],
        wu0,
        wl0,
        wd0,
        s0,
        fr0,
    )


# ---------------------------------------------------------------------------
# Routing (CSL + linear reservoir, PBM-aligned)
# ---------------------------------------------------------------------------


def route_csl_qs_from_qt(qt, cs, l):
    """CSL lag routing; stack-based recurrence (autograd-safe)."""
    n_time, n_basin = qt.shape
    cols = []
    for j in range(n_basin):
        lag = int(l[j].item())
        if lag < 0:
            lag = 0
        effective_lag = min(lag, n_time - 1)
        steps = []
        for t in range(n_time):
            if t < effective_lag:
                steps.append(qt[t, j])
            else:
                q_prev = steps[t - 1]
                steps.append(
                    cs[j] * q_prev + (1.0 - cs[j]) * qt[t - effective_lag, j]
                )
        cols.append(torch.stack(steps, dim=0))
    return torch.stack(cols, dim=1)


def route_qt_reservoir(rss, ris, rgs, runoff_im, ci, cg, qi0, qg0):
    n_time = rss.shape[0]
    qi = qi0.clone()
    qg = qg0.clone()
    steps = []
    for t in range(n_time):
        qi = ci * qi + (1.0 - ci) * ris[t]
        qg = cg * qg + (1.0 - cg) * rgs[t]
        steps.append(rss[t] + runoff_im[t] + qi + qg)
    return torch.stack(steps, dim=0)


def route_csl(rss, ris, rgs, runoff_im, cs, l, ci, cg, qi0, qg0):
    qt = route_qt_reservoir(rss, ris, rgs, runoff_im, ci, cg, qi0, qg0)
    return route_csl_qs_from_qt(qt, cs, l)


def route_linear(ris, rgs, qs_surface, ci, cg, qi0, qg0):
    n_time = ris.shape[0]
    qi = qi0.clone()
    qg = qg0.clone()
    steps = []
    for t in range(n_time):
        qi = ci * qi + (1.0 - ci) * ris[t]
        qg = cg * qg + (1.0 - cg) * rgs[t]
        steps.append(qs_surface[t] + qi + qg)
    return torch.stack(steps, dim=0)


def route_linear_grad_from(
    ris,
    rgs,
    qs_surface,
    ci,
    cg,
    qi0,
    qg0,
    *,
    grad_from_t,
):
    n_time, n_basin = ris.shape
    t0 = int(grad_from_t)
    if t0 <= 0:
        return route_linear(ris, rgs, qs_surface, ci, cg, qi0, qg0)

    qi, qg = qi0, qg0
    prefix = []
    if t0 > 0:
        with torch.no_grad():
            for t in range(t0):
                qi = ci * qi + (1.0 - ci) * ris[t]
                qg = cg * qg + (1.0 - cg) * rgs[t]
                prefix.append(qs_surface[t] + qi + qg)

    post = []
    for t in range(t0, n_time):
        qi = ci * qi + (1.0 - ci) * ris[t]
        qg = cg * qg + (1.0 - cg) * rgs[t]
        post.append(qs_surface[t] + qi + qg)

    if prefix:
        return torch.cat([torch.stack(prefix, dim=0), torch.stack(post, dim=0)], dim=0)
    return torch.stack(post, dim=0)


def route_csl_grad_from(
    rss,
    ris,
    rgs,
    runoff_im,
    cs,
    l,
    ci,
    cg,
    qi0,
    qg0,
    *,
    grad_from_t,
):
    t0 = int(grad_from_t)
    if t0 <= 0:
        return route_csl(rss, ris, rgs, runoff_im, cs, l, ci, cg, qi0, qg0)

    n_time = rss.shape[0]
    qi, qg = qi0, qg0
    qt_steps = []
    for t in range(n_time):
        if t < t0:
            with torch.no_grad():
                qi = ci * qi + (1.0 - ci) * ris[t]
                qg = cg * qg + (1.0 - cg) * rgs[t]
                qt_steps.append(rss[t] + runoff_im[t] + qi + qg)
        else:
            qi = ci * qi + (1.0 - ci) * ris[t]
            qg = cg * qg + (1.0 - cg) * rgs[t]
            qt_steps.append(rss[t] + runoff_im[t] + qi + qg)
    qt = torch.stack(qt_steps, dim=0)
    return route_csl_qs_from_qt(qt, cs, l)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def simulate_dXAJ(
    p_and_e,
    params,
    *,
    warmup_length=365,
    grad_steps=None,
):
    dtype = p_and_e.dtype
    device = p_and_e.device
    params = constrain_ki_kg(params.to(dtype=dtype))
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

    wu0, wl0, wd0, s0 = initial_storage_from_forcing(
        p_and_e, um, lm, dm, sm, warmup_length=warmup_length
    )
    n_basin = p_and_e.shape[1]
    fr0 = p_and_e.new_full((n_basin,), 0.1)
    qi0 = p_and_e.new_full((n_basin,), 0.1)
    qg0 = p_and_e.new_full((n_basin,), 0.1)

    n_time = p_and_e.shape[0]
    steps = effective_grad_steps(grad_steps, n_time)
    if steps >= n_time:
        runoff_im, rss, ris, rgs, _, _, _, _, _ = scan_runoff_series(
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
        )
        return route_csl(rss, ris, rgs, runoff_im, cs, l, ci, cg, qi0, qg0)

    t0 = n_time - steps
    (
        runoff_im_pre,
        rss_pre,
        ris_pre,
        rgs_pre,
        wu_mid,
        wl_mid,
        wd_mid,
        s_mid,
        fr_mid,
    ) = prefix_runoff_numba(
        p_and_e[:t0], params, wu0, wl0, wd0, s0, fr0, device, dtype
    )
    (
        runoff_im_post,
        rss_post,
        ris_post,
        rgs_post,
        _,
        _,
        _,
        _,
        _,
    ) = scan_runoff_series(
        p_and_e[t0:],
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
        wu_mid.detach(),
        wl_mid.detach(),
        wd_mid.detach(),
        s_mid.detach(),
        fr_mid.detach(),
    )

    runoff_im = torch.cat([runoff_im_pre.detach(), runoff_im_post], dim=0)
    rss = torch.cat([rss_pre.detach(), rss_post], dim=0)
    ris = torch.cat([ris_pre.detach(), ris_post], dim=0)
    rgs = torch.cat([rgs_pre.detach(), rgs_post], dim=0)

    return route_csl_grad_from(
        rss,
        ris,
        rgs,
        runoff_im,
        cs,
        l,
        ci,
        cg,
        qi0,
        qg0,
        grad_from_t=t0,
    )


class XAJCore(nn.Module):
    """Differentiable XAJ with CSL routing (P, PET drivers)."""

    driver_dim = 2
    pbm_name = 'dXAJ'

    def __init__(
        self,
        warmup_length=365,
        *,
        pbm_grad_steps=None,
    ):
        super().__init__()
        self.warmup_length = int(warmup_length)
        self.pbm_grad_steps = pbm_grad_steps

    def forward(self, drivers, params):
        if drivers.shape[-1] != self.driver_dim:
            raise ValueError(
                f'XAJCore expects {self.driver_dim} drivers, got {drivers.shape[-1]}'
            )
        qsim = simulate_dXAJ(
            drivers,
            params,
            warmup_length=self.warmup_length,
            grad_steps=self.pbm_grad_steps,
        )
        return qsim.unsqueeze(-1)
