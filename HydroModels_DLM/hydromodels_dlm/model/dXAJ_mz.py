"""Differentiable XAJ-mz (gamma unit hydrograph routing), aligned with HydroModels_PBM."""

import torch
from torch import nn

from hydromodels_dlm.model.dGR4J import convolve_truncated_batched, effective_grad_steps
from hydromodels_dlm.model.dXAJ import (
    constrain_ki_kg,
    initial_storage_from_forcing,
    prefix_runoff_numba,
    route_linear,
    route_linear_grad_from,
    scan_runoff_series,
)


def gamma_uh_weights_batched(a, theta, len_uh):
    aa = torch.clamp(a, min=0.0) + 0.1
    th = torch.clamp(theta, min=0.0) + 0.5
    t = torch.arange(0.5, 0.5 + float(len_uh), device=a.device, dtype=a.dtype).view(1, -1)
    aa_col = aa.view(-1, 1)
    th_col = th.view(-1, 1)
    denom = torch.exp(torch.lgamma(aa_col)) * (th_col ** aa_col)
    w = (t ** (aa_col - 1.0)) * torch.exp(-t / th_col) / denom
    return w / w.sum(dim=-1, keepdim=True)


def route_mz(runoff_im, rss, ris, rgs, a, theta, ci, cg, qi0, qg0, kernel_size):
    n_time, n_basin = ris.shape
    effective_len = min(int(kernel_size), n_time)
    inp = runoff_im + rss
    w = gamma_uh_weights_batched(a, theta, effective_len)
    qs_surface = convolve_truncated_batched(inp, w)
    return route_linear(ris, rgs, qs_surface, ci, cg, qi0, qg0)


def route_mz_grad_from(
    runoff_im,
    rss,
    ris,
    rgs,
    a,
    theta,
    ci,
    cg,
    qi0,
    qg0,
    kernel_size,
    *,
    grad_from_t,
):
    n_time, n_basin = ris.shape
    t0 = int(grad_from_t)
    effective_len = min(int(kernel_size), n_time)
    inp = runoff_im + rss
    w = gamma_uh_weights_batched(a, theta, effective_len)
    qs_surface = convolve_truncated_batched(inp, w)
    return route_linear_grad_from(
        ris,
        rgs,
        qs_surface,
        ci,
        cg,
        qi0,
        qg0,
        grad_from_t=t0,
    )


def simulate_dXAJ_mz(
    p_and_e,
    params,
    *,
    warmup_length=365,
    kernel_size=15,
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
    a = params[:, 11]
    theta = params[:, 12]
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
        return route_mz(
            runoff_im, rss, ris, rgs, a, theta, ci, cg, qi0, qg0, kernel_size
        )

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
    return route_mz_grad_from(
        runoff_im,
        rss,
        ris,
        rgs,
        a,
        theta,
        ci,
        cg,
        qi0,
        qg0,
        kernel_size,
        grad_from_t=t0,
    )


# ---------------------------------------------------------------------------
# PBM core
# ---------------------------------------------------------------------------


class XAJMzCore(nn.Module):
    """Differentiable XAJ-mz (P, PET drivers)."""

    driver_dim = 2
    pbm_name = 'dXAJ-mz'

    def __init__(
        self,
        warmup_length=365,
        *,
        kernel_size=15,
        pbm_grad_steps=None,
    ):
        super().__init__()
        self.warmup_length = int(warmup_length)
        self.kernel_size = int(kernel_size)
        self.pbm_grad_steps = pbm_grad_steps

    def forward(self, drivers, params):
        if drivers.shape[-1] != self.driver_dim:
            raise ValueError(
                f'XAJMzCore expects {self.driver_dim} drivers, got {drivers.shape[-1]}'
            )
        qsim = simulate_dXAJ_mz(
            drivers,
            params,
            warmup_length=self.warmup_length,
            kernel_size=self.kernel_size,
            grad_steps=self.pbm_grad_steps,
        )
        return qsim.unsqueeze(-1)
